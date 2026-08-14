import torch.nn as nn

from .Attention import LayerNorm, PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, trace, log):
        input_norm = self.layer_norm(inputs)
        context, attn = self.self_attn(input_norm, trace, log)

        out = self.dropout(context) + inputs
        return self.feed_forward(out), attn

class CrossTransformer(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, attn_modules):
        super(CrossTransformer, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList([TransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, metric, trace, log):
        '''
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        :return:
        '''

        out = metric
        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, trace, log)
        out = self.layer_norm(out)

        return out

