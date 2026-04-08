"""
Loss functions for SpliceMamba.

Handles the extreme class imbalance (~6200:1 neither:splice) with
per-class alpha weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with per-class alpha weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : focusing parameter (default 2.0)
    alpha : per-class weights as list/tensor of length C
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, C) raw logits — NOT softmax'd
        targets : (N,) int64 class indices

        Returns
        -------
        scalar focal loss (mean over N positions)
        """
        # Numerically stable log-softmax
        log_probs = F.log_softmax(logits, dim=-1)       # (N, C)
        probs = log_probs.exp()                          # (N, C)

        # Gather the true-class probabilities
        targets_unsq = targets.unsqueeze(-1)             # (N, 1)
        log_pt = log_probs.gather(dim=-1, index=targets_unsq).squeeze(-1)  # (N,)
        pt = probs.gather(dim=-1, index=targets_unsq).squeeze(-1)          # (N,)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma            # (N,)

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]                 # (N,)
        else:
            alpha_t = 1.0

        loss = -alpha_t * focal_weight * log_pt           # (N,)
        return loss.mean()


class WeightedCE(nn.Module):
    """Weighted cross-entropy loss with per-class alpha weights.

    Simpler than focal loss — no (1-p_t)^gamma focusing term.
    Better calibrated outputs: the model is penalized equally for all
    misclassifications within a class, not just hard examples.

    Parameters
    ----------
    alpha : per-class weights as list/tensor of length C
    """

    def __init__(self, alpha: list[float]):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, C) raw logits
        targets : (N,) int64 class indices

        Returns
        -------
        scalar weighted cross-entropy loss (mean over N)
        """
        return F.cross_entropy(logits, targets, weight=self.alpha)
