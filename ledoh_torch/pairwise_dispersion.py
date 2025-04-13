import torch
from torch import Tensor
import torch.nn.functional as F
from .sphere_dispersion import SphereDispersion

GEODESIC_EPS = 1e-4

@torch.jit.script
def get_batched_dot_product(X: Tensor, batch_size: int):
    if batch_size < 0:
        X_batch = X
    else:
        batch_idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
        X_batch = torch.index_select(X, 0, batch_idx)

    similarities = X_batch @ X_batch.T
    return similarities


### Kernels on sphere from cosines.

def _k_gauss_eucl(cosines: Tensor, gamma: float = 1):
    # truly, the correct case is -2 here but it basically just changes the
    # regularizer strength by exp(-gamma*2).
    # negative_dist_sq = cosines - 2
    negative_dist_sq = cosines
    return torch.exp(gamma * negative_dist_sq)


def _k_lap_eucl(cosines: Tensor, gamma: float = 1):
    negative_dist = -torch.sqrt(2 - cosines)
    return torch.exp(gamma * negative_dist)


def _k_gauss_geo(cosines: Tensor, gamma: float = 1):
    negative_dist_sq = -(torch.acos(cosines) ** 2)
    return torch.exp(gamma * negative_dist_sq)


def _k_lap_geo(cosines: Tensor, gamma: float = 1):
    cosines = torch.clip(cosines, -1 + GEODESIC_EPS, 1 - GEODESIC_EPS)
    negative_dist = -torch.acos(cosines)
    return torch.exp(gamma * negative_dist)


def _k_riesz_eucl(cosines: Tensor, s: float = 1):
    dist_sq = 2-cosines

    if s > 0:
        return dist_sq ** -(s/2)
    elif s == 0:  # log(1/dist)
        return -torch.log(dist_sq) / 2
    elif s < 0:
        return -(dist_sq ** -(s/2))


def _k_riesz_geo(cosines: Tensor, s: float = 1):
    cosines = torch.clip(cosines, -1 + GEODESIC_EPS, 1 - GEODESIC_EPS)
    dist = torch.acos(cosines)

    if s > 0:
        return dist ** -s
    elif s == 0:  # log(1/dist)
        return -torch.log(dist)
    elif s < 0:
        return -(dist ** -s)


KERNELS = {
    ("gaussian", "euclidean"): _k_gauss_eucl,
    ("gaussian", "geodesic"): _k_gauss_geo,
    ("laplace", "euclidean"): _k_lap_eucl,
    ("laplace", "geodesic"): _k_lap_geo,
    ("riesz", "euclidean"): _k_riesz_eucl,
    ("riesz", "geodesic"): _k_riesz_geo,
}


class PairwiseDispersion(SphereDispersion):
    def __init__(self, batch_size: int = -1, eps:float = 1e-7):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps


class KernelSphereDispersion(PairwiseDispersion):
    def __init__(self,
                 kernel: str,
                 distance: str,
                 kernel_args: dict,
                 batch_size: int = -1,
        ):
        """Dispersion of a set of points on the sphere using kernel function.
           :param batch_size: reduce the memory usage by computing the dispersion on a batch of size batch_size
           :return: loss value, {"sample_size": batch_size}
        """
        super().__init__(batch_size)
        if batch_size != -1:
            raise ValueError("batch size is deprecated; sample outside the regularizer call")
        self.kernel_args = kernel_args
        try:
            self._kernel = KERNELS[kernel, distance]
        except KeyError:
            raise ValueError("Unsupported kernel and distance configuration. "
                             "Must be one of " + str(KERNELS.keys()))

    def _cosines(self, X: Tensor):
        all_cosines = get_batched_dot_product(X, self.batch_size)
        n = all_cosines.shape[0]
        return all_cosines[*torch.triu_indices(n, n, offset=1)]

    def forward(self, X:Tensor) -> Tensor:
        assert(not X.isnan().any())
        cosines = self._cosines(X)
        assert(not cosines.isnan().any())
        kervals = self._kernel(cosines, **self.kernel_args)
        assert(not kervals.isnan().any())
        return kervals.mean()


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


# Below is deprecated in favor of KernelSphereDispersion with riesz
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

        return -loss.min(dim=1)[0].mean()
