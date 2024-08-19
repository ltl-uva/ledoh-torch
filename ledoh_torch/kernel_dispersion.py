import torch
from torch import Tensor

from .sphere_dispersion import SphereDispersion


class KernelSphereDispersion(SphereDispersion):
    def __init__(self, gamma: float = 1.0, batch_size: int = -1):
        """Dispersion of a set of points on the sphere using kernel function.
           :param gamma: scaling factor
           :param batch_size: reduce the memory usage by computing the dispersion on a batch of size batch_size
           :return: loss value, {"sample_size": batch_size}
        """
        super().__init__()
        self.gamma = gamma
        self.batch_size = batch_size

    def forward(self, X: Tensor) -> Tensor:

        batch_size = X.shape[0] if self.batch_size < 0 else self.batch_size
        print(batch_size)

        batch_idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
        X_batch = torch.index_select(X, 0, batch_idx)

        similarities = X_batch @ X_batch.T
        similarities = torch.triu(similarities)

        loss = torch.exp(self.gamma * similarities).sum() * (2.0 / (batch_size * (batch_size - 1)))
        return loss
