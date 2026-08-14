import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.functional import F


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, feat_drop=0.3, aggregator_type='mean', num_layers=2, use_residual=True, norm_type='batch'):
        super(SAGEEncoder, self).__init__()

        self.num_layers=num_layers
        self.use_residual = use_residual
        self.norm_type = norm_type

        self.sage_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        self.activation = F.relu

        if num_layers == 1:
            self.sage_layers.append(
                dglnn.SAGEConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    feat_drop=feat_drop,
                    aggregator_type=aggregator_type,
                    activation=None,
                    bias=False
                )
            )
            if use_residual and in_dim != out_dim:
                self.residual_proj.append(
                    nn.Linear(in_dim, out_dim, bias=False)
                )
            else:
                self.residual_proj.append(nn.Identity())

            if norm_type == 'batch':
                self.norm_layers.append(nn.BatchNorm1d(out_dim))
            elif norm_type == 'layer':
                self.norm_layers.append(nn.LayerNorm(out_dim))
            else:
                self.norm_layers.append(nn.Identity())

        else:
            self.sage_layers.append(
                dglnn.SAGEConv(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    feat_drop=feat_drop,
                    activation=self.activation,
                    aggregator_type=aggregator_type,
                    bias=False
                )
            )
            if use_residual and in_dim != hidden_dim:
                self.residual_proj.append(
                    nn.Linear(in_dim, hidden_dim, bias=False)
                )
            else:
                self.residual_proj.append(nn.Identity())

            if norm_type == 'batch':
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == 'layer':
                self.norm_layers.append(nn.LayerNorm(hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())

            for l in range(0, num_layers-2):
                self.sage_layers.append(
                    dglnn.SAGEConv(
                        in_feats=hidden_dim,
                        out_feats=hidden_dim,
                        feat_drop=feat_drop,
                        aggregator_type=aggregator_type,
                        activation=self.activation,
                        bias=False
                    )
                )
                if use_residual and hidden_dim != hidden_dim:
                    self.residual_proj.append(
                        nn.Linear(hidden_dim, hidden_dim, bias=False)
                    )
                else:
                    self.residual_proj.append(nn.Identity())

                if norm_type == 'batch':
                    self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
                elif norm_type == 'layer':
                    self.norm_layers.append(nn.LayerNorm(hidden_dim))
                else:
                    self.norm_layers.append(nn.Identity())

            self.sage_layers.append(
                dglnn.SAGEConv(
                    in_feats=hidden_dim,
                    out_feats=out_dim,
                    feat_drop=feat_drop,
                    aggregator_type=aggregator_type,
                    activation=None,
                    bias=False
                )
            )
            if use_residual and hidden_dim != out_dim:
                self.residual_proj.append(nn.Linear(hidden_dim, out_dim, bias=False))
            else:
                self.residual_proj.append(nn.Identity())

            if norm_type == 'batch':
                self.norm_layers.append(nn.BatchNorm1d(out_dim))
            elif norm_type == 'layer':
                self.norm_layers.append(nn.LayerNorm(out_dim))
            else:
                self.norm_layers.append(nn.Identity())

    def forward(self, batch_graphs, x):
        h = x
        h_prev = h
        for l in range(self.num_layers):
            residual = h_prev

            h = self.sage_layers[l](batch_graphs, h)

            if self.use_residual:
                if residual.shape[-1] != h.shape[-1]:
                    residual = self.residual_proj[l](residual)
                h = h + residual
            h = self.norm_layers[l](h)
            if l < self.num_layers - 1:
                h = self.activation(h)
            h_prev = h
        return h
