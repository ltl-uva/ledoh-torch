from typing import Tuple, Dict, Any

import torch
from torch import Tensor

from .sphere_dispersion import SphereDispersion

class KernelSphereDispersion(SphereDispersion):
    @staticmethod
    def forward(X: Tensor, reduction="mean", gamma: float=0.001, batch_size: int =-1)->Tuple[Tensor, Dict[str, Any]]:
        """Compute the dispersion of a set of points on the sphere using kernel function.
        :param X: points on the sphere
        :param gamma: scaling factor
        :param batch_size: reduce the memory usage by computing the dispersion on a batch of size batch_size
        :return: loss value, {"sample_size": batch_size}
        """
        if batch_size < 0:
            batch_size = X.shape[0]

        batch_idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
        X_batch = torch.index_select(X, 0, batch_idx)

        similarities = X_batch @ X_batch.T
        similarities.fill_diagonal_(0)

        loss = torch.exp(gamma * similarities).sum()
        if reduction == "mean":
            loss = torch.div(loss, batch_size)
            return loss, {"sample_size": 1}
        else:
            return loss, {"sample_size": batch_size}
