import torch
from torch import Tensor

from .sphere_dispersion import SphereDispersion


class KernelSphereSemibatchDispersion(SphereDispersion):
    @staticmethod
    def forward(X: Tensor,
                gamma: float = 0.001,
                batch_size: int = -1) -> Tensor:
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

        # size [n, batch size]
        similarities = X @ X_batch.T
        similarities.fill_diagonal_(0)
        norm_const = similarities.numel() - min(similarities.shape) # number of non-diagonal entries
        loss = torch.exp(gamma * similarities).sum() / norm_const
        return loss

