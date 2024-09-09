"""
Implementation of Maximum Minimum Pairwise Angles (MMA) loss function
proposed in https://doi.org/10.48550/arXiv.2006.06527.
"""

import torch
from torch import Tensor

from .sphere_dispersion import SphereDispersion


class MMADispersion(SphereDispersion):
    def __init__(self, batch_size: int = -1, eps=1e-7):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps

    def forward(self, X: Tensor) -> Tensor:

        batch_size = X.shape[0] if self.batch_size < 0 else self.batch_size

        batch_idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
        X_batch = torch.index_select(X, 0, batch_idx)

        similarities = X_batch @ X_batch.T
        angles = torch.arccos(similarities.clamp(-1 + self.eps, 1 - self.eps))
        angles = angles.fill_diagonal_(torch.inf)
        return -angles.min(dim=1)[0].mean()

