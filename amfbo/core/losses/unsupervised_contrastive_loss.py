import torch
import torch.nn.functional as F
from torch import nn


class UspConLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.device = device

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        logits = z_i @ z_j.T
        logits /= self.temperature
        n = z_j.shape[0]
        labels = torch.arange(0, n, dtype=torch.long).cuda()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss


# contrastive_loss = UspConLoss(3, 1.0)
#
# I = torch.tensor([[1.0, 2.0], [3.0, -2.0], [1.0, 5.0]], requires_grad=True)
# # J = torch.tensor([[1.0, 0.75], [2.8, -1.75], [1.0, 4.7]], requires_grad=True)
# J = torch.tensor([[1.0, 1.75], [2.8, -1.75], [1.0, 4.7]],requires_grad=True)
#
# loss = contrastive_loss(I, J)
#
# print(loss)
