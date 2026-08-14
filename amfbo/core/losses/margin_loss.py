import torch
import torch.nn as nn


class GaussianMarginLoss(nn.Module):
    def __init__(self, delta=0.2, sigma=0.1, eps=1e-8, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.sigma = sigma
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds):
        max_values, _ = torch.max(preds, dim=1, keepdim=True)
        violations = torch.clamp(preds - (max_values - self.delta), min=0)

        _, max_indices = torch.max(preds, dim=1, keepdim=True)
        mask = torch.ones_like(preds, dtype=torch.bool)
        mask.scatter_(1, max_indices, False)

        distances = max_values - preds + self.eps
        weights = torch.exp(-distances.pow(2) / self.sigma ** 2)

        weighted_violations = violations * weights * mask.float()
        loss = weighted_violations.sum(dim=1)

        return self._reduce_loss(loss)

    def _reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
