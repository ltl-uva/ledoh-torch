import torch
from torch import Tensor
import torch.nn.functional as F
from .sphere_dispersion import SphereDispersion

@torch.jit.script
def get_batched_dot_product(X: Tensor, batch_size: int):
    batch_size = X.shape[0] if batch_size < 0 else batch_size

    batch_idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
    X_batch = torch.index_select(X, 0, batch_idx)

    similarities = X_batch @ X_batch.T
    return similarities

class PairwiseDispersion(SphereDispersion):
    def __init__(self, batch_size: int = -1, eps:float = 1e-7):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps

class KernelSphereDispersion(PairwiseDispersion):
    def __init__(self, gamma: float = 1.0, batch_size: int = -1):
        """Dispersion of a set of points on the sphere using kernel function.
           :param gamma: scaling factor
           :param batch_size: reduce the memory usage by computing the dispersion on a batch of size batch_size
           :return: loss value, {"sample_size": batch_size}
        """
        super().__init__(batch_size)
        self.gamma = gamma

    def forward(self, X:Tensor) -> Tensor:
        similarities = get_batched_dot_product(X, self.batch_size)
        similarities = similarities-torch.tril(similarities)
        cnt = 2.0 / (self.batch_size * (self.batch_size - 1))
        loss = torch.exp(self.gamma * similarities).sum() * cnt
        return loss

class MMADispersion(PairwiseDispersion):
    def __init__(self, batch_size: int = -1, eps=1e-7):
        super().__init__(batch_size=batch_size, eps=eps)

    def forward(self, X: Tensor) -> Tensor:
        similarities = get_batched_dot_product(X, self.batch_size)
        angles = torch.arccos(similarities.clamp(-1 + self.eps, 1 - self.eps))
        angles = angles.fill_diagonal_(torch.inf)
        return -angles.min(dim=1)[0].mean()

class MMCSDispersion(PairwiseDispersion):
    def forward(self, X: Tensor) -> Tensor:
        similarities = get_batched_dot_product(X, self.batch_size) + 1
        similarities -= 2. * torch.diag(torch.diag(similarities))
        loss = similarities.max(dim=1)[0]
        return loss.mean()

class MHEDispersion(PairwiseDispersion):
    def __init__(self, batch_size: int = -1, eps=1e-7, s_power=0):
        super().__init__(batch_size=batch_size, eps=eps)
        self.s_power = s_power

    def forward(self, X: Tensor) -> Tensor:
        similarities = get_batched_dot_product(X, self.batch_size)
        angles = torch.arccos(similarities.clamp(-1 + self.eps, 1 - self.eps))

        loss = 0
        if self.s_power == 0:
            loss = -torch.log(angles)

        if self.s_power>0:
            loss = (torch.pow(angles, torch.ones_like(angles)*-self.s_power))

        loss = loss - torch.tril(loss)
        cnt = 2.0 / (self.batch_size * (self.batch_size - 1))
        loss = loss.sum() * cnt
        return loss


class KoLeoDispersion(PairwiseDispersion):
    def forward(self, X: Tensor) -> Tensor:
        pdist = torch.cdist(X, X, p=2.0)
        loss = torch.log(pdist)
        loss = loss.fill_diagonal_(torch.inf)

        return loss.min(dim=1)[0].mean()


