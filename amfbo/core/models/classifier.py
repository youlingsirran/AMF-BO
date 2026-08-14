import torch.nn as nn

from amfbo.core.models.backbone.fc import FullyConnected


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, device='cpu'):
        super(Classifier, self).__init__()

        self.net = nn.Linear(in_dim, out_dim).to(device)

    def forward(self, h):
        return self.net(h)

class Classifier2(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[64, 128], device='cpu'):
        super(Classifier2, self).__init__()

        self.net = FullyConnected(in_dim, out_dim, hiddens).to(device)

    def forward(self, h):
        return self.net(h)