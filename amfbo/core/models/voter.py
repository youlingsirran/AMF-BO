
import torch.nn as nn

from amfbo.core.models.backbone.fc import FullyConnected


class Voter(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[64, 128], device='cpu'):
        super(Voter, self).__init__()

        self.net = FullyConnected(in_dim, out_dim, hiddens).to(device)

    def forward(self, h):
        return self.net(h)

class Voter2(nn.Module):
    def __init__(self, in_dim, out_dim, device='cpu'):
        super(Voter2, self).__init__()

        self.net = nn.Linear(in_dim, out_dim).to(device)

    def forward(self, h):
        return self.net(h)