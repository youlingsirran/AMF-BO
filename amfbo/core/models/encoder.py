import torch
import torch.nn.functional as F
from torch import nn

from amfbo.core.models.backbone.dominant import (
    compute_entropy_batch_parallel, entropy_vec, feature_statistics,
    geometric_weighted_fusion, process_columns, signed_harmonic_fusion)
from amfbo.core.models.backbone.sage import SAGEEncoder
from amfbo.core.models.classifier import Classifier, Classifier2
from amfbo.core.models.fusion.attention import (AdaptiveFusionTopology,
                                                TripleCrossAttnFusion,
                                                TripleFusion,
                                                TripleGuidedFusion4Head,
                                                anomaly_topk_pooling,
                                                cal_rcl_loss)
from amfbo.core.models.voter import Voter, Voter2


class Encoder(nn.Module):
    def __init__(self, 
                 alert_embedding_dim: int,
                 graph_hidden_dim: int,
                 graph_out_dim: int,
                 num_layers=3,
                 aggregator='mean',
                 feat_drop=0.3,
                 ft_num=5,
                 bins=20,
                 alpha=0.5):
        super(Encoder, self).__init__()

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.bins = bins
        self.alpha = alpha
        self.graph_encoder = nn.ModuleDict()
        for modality in ['metric', 'log', 'trace']:
            self.graph_encoder[modality] = SAGEEncoder(
                in_dim=128,
                out_dim=32,
                hidden_dim=64,
                num_layers=num_layers,
                aggregator_type=aggregator,
                feat_drop=feat_drop
            )
        # self.fusion = TripleCrossAttnFusion(32)
        self.fusion = AdaptiveFusionTopology(32)
        # # self.fusion = TripleGuidedFusion(64)
        # # self.fusion = TripleGuidedFusion4Head(32)
        # self.fusion = ChunkBasedSequenceFusion(64, 1,4)

        self.locator = Voter(32,
                             hiddens=[16],
                             out_dim=1)
        self.locator2 = Voter(32,
                             hiddens=[16],
                             out_dim=1)

        self.typeClassifier = Classifier(in_dim=32,
                                         out_dim=ft_num)

        self.pre_abc = nn.Sequential(
            nn.Linear(96, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.pre_ab = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

    def forward(self, batch_graphs):

        fs, es = {}, {}
        for modality, encoder in self.graph_encoder.items():
            x_d = batch_graphs.ndata[modality]
            e_d = encoder(batch_graphs, x_d)
            es[modality] = e_d
        metric = es['metric']
        log = es['log']
        trace = es['trace']
        # metric_word = batch_graphs.ndata['metric_word']
        # log_word = batch_graphs.ndata['log_word']
        # trace_word = batch_graphs.ndata['trace_word']
        # metric, log, trace = torch.cat([metric, metric_word], dim=1), torch.cat([log, log_word], dim=1), torch.cat(
        #     [trace, trace_word], dim=1)
        F23 = torch.cat([log, trace], dim=1)
        F13 = torch.cat([metric, trace], dim=1)
        F12 = torch.cat([metric, log], dim=1)
        F123 = torch.cat([metric, log, trace], dim=1)

        A1 = compute_entropy_batch_parallel(F123, bins=self.bins) - compute_entropy_batch_parallel(F23, bins=self.bins)
        A2 = compute_entropy_batch_parallel(F123, bins=self.bins) - compute_entropy_batch_parallel(F13, bins=self.bins)
        A3 = compute_entropy_batch_parallel(F123, bins=self.bins) - compute_entropy_batch_parallel(F12, bins=self.bins)

        entropy_batch1 = torch.stack([A1, A2, A3], dim=0).transpose(0, 1)
        entropy_batch1 = F.softmax(entropy_batch1, dim=-1)

        B1 = 1 - F.cosine_similarity(self.pre_abc(F123), self.pre_ab(F23), dim=-1)
        B2 = 1 - F.cosine_similarity(self.pre_abc(F123), self.pre_ab(F13), dim=-1)
        B3 = 1 - F.cosine_similarity(self.pre_abc(F123), self.pre_ab(F12), dim=-1)
        entropy_batch2 = torch.stack([B1, B2, B3], dim=0).transpose(0, 1)
        entropy_batch2 = F.softmax(entropy_batch2, dim=-1)

        entropy_batch = geometric_weighted_fusion(entropy_batch1, entropy_batch2, alpha=self.alpha)

        common_emb = self.fusion(metric, log, trace, entropy_batch)

        root_logit = self.locator(common_emb)
        l_rcl, anomaly_scores = cal_rcl_loss(root_logit, batch_graphs)
        f = anomaly_topk_pooling(batch_graphs, common_emb, anomaly_scores, K=6)
        type_logit = self.typeClassifier(f)
        max_counts, importance_sums = process_columns(entropy_batch)
        return root_logit, type_logit, max_counts, importance_sums, l_rcl
