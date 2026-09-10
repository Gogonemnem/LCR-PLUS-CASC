"""Shared loss functions for the CASC and LCR classifiers."""
import torch
from torch import nn
import torch.nn.functional as F


class LQLoss(nn.Module):
    """Generalized cross-entropy (GCE) loss with per-class weights, applied to probabilities."""

    def __init__(self, q, weight, alpha=0.0):
        super().__init__()
        self.q = q
        self.alpha = alpha
        weight = torch.log(1 / torch.as_tensor(weight, dtype=torch.float32))
        self.register_buffer('weight', F.softmax(weight, dim=-1))

    def forward(self, probs, target):
        # probs: [B, C] class probabilities
        yq = torch.gather(probs, 1, target.unsqueeze(1))
        lq = (1 - yq ** self.q) / self.q
        weight = torch.gather(self.weight.expand_as(probs), 1, target.unsqueeze(1))
        return (self.alpha * lq + (1 - self.alpha) * lq * weight).mean()

    def set_weights(self, weights):
        weight = torch.log(1 / torch.as_tensor(weights, dtype=torch.float32, device=self.weight.device))
        self.weight.copy_(F.softmax(weight, dim=-1))


class GCEQ(nn.Module):
    """Generalized cross entropy with exponent q, applied on the true-class probability."""

    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, logits, y_true):
        # logits: raw logits [B, C]; y_true: class indices [B]
        log_p = F.log_softmax(logits, dim=1)
        p_true = torch.gather(log_p, 1, y_true.unsqueeze(1)).squeeze(1).exp()
        return ((1 - p_true ** self.q) / self.q).mean()
