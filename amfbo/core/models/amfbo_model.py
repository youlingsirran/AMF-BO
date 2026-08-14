from torch import nn

from amfbo.config.experiment_config import Config
from amfbo.core.models.encoder import Encoder


class AMFBOModel(nn.Module):
    def __init__(self, config: Config):
        super(AMFBOModel, self).__init__()

        self.encoder = Encoder(
                alert_embedding_dim=config.alert_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,   # mean
                feat_drop=config.feat_drop,
                ft_num=config.ft_num,
                bins=config.bins,
                alpha=config.alpha
            )

    def forward(self, batch_graphs):

        root_logit, type_logit, max_counts, importance_sums, l_rcl = self.encoder(batch_graphs)
        return root_logit, type_logit, max_counts, importance_sums, l_rcl
