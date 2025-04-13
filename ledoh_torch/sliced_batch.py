import math
from typing import Optional

import torch
from torch import Tensor

# from .utils import init_great_circle
from .sphere_dispersion import SphereDispersion


class AxisAlignedBatchSphereDispersion(SphereDispersion):
    """Wrap sliced_batch to provide consistent API"""
    def __init__(self, n_samples: int = 1, batch_size: int = -1):
        super().__init__()
        self.n_samples = n_samples
        if batch_size != -1:
            raise ValueError("deprecated, name is misleading")

    def forward(self, X:Tensor):
        # this is based on a misunderstanding i think.
        # we should definitely not default to n samples here.
        # batch_size = X.shape[0] if self.batch_size < 0 else self.batch_size
        return sliced_batch(X, n_samples=self.n_samples)


def sliced_batch(X, n_samples=5):
    n, d = X.shape
    device = X.device

    # generate random axes quickly
    i = torch.randint(low=0, high=d, size=(n_samples,))
    j = torch.randint(low=0, high=d-1, size=(n_samples,))
    j[j >= i] += 1
    assert (i != j).all()

    # shape: [n, s] both
    Xp = X[:, i]
    Xq = X[:, j]

    thetas = torch.arctan2(Xq, Xp)
    ix = torch.argsort(thetas, dim=0)  # each ix[:, k] is a perm


    phis = 2 * math.pi * torch.arange(1, n+1) / n
    phis -= math.pi + math.pi / n  # make zero-mean
    phis = phis.to(device)

    # apply the inverse of each ix[:, k] permutation to the phis
    # https://discuss.pytorch.org/t/how-to-quickly-inverse-a-permutation-by-using-pytorch/116205/7
    theta_star = torch.empty_like(thetas)
    theta_star.scatter_(dim=0, index=ix, src=phis.unsqueeze(1).expand(-1, n))

    theta_star += torch.mean(thetas, dim=0, keepdims=True)

    return .5 * torch.sum(torch.square(thetas - theta_star)) / n_samples


if __name__ == '__main__':
    sliced_batch(torch.randn(10, 30), n_samples=15)

