import math
from typing import Optional

import torch
from torch import Tensor

from .utils import init_great_circle
from .sphere_dispersion import SphereDispersion


def _dist_along_slice(Xp, Xq, N, device):
    thetas = torch.arctan2(Xq, Xp)

    ix = torch.argsort(thetas)
    invix = torch.empty_like(ix)
    invix[ix] = torch.arange(N, device=device)

    phis = 2 * math.pi * torch.arange(1, N+1, device=device) / N
    phis -= math.pi + math.pi / N  # make zero-mean
    phis = phis[invix]

    thetas_star = torch.mean(thetas) + phis

    return .5 * torch.sum(torch.square(thetas - thetas_star))


class SlicedSphereDispersion(SphereDispersion):

    def forward(self, X: Tensor) -> Tensor:
        """
        calculates forward pass for sliced dispersion
        :param reduction:
        :param return_hidden_states: return values to calculate grad
        :param X: Tensor of shape (N, d)
        :param p, q: vectors defining great circle. p is orthogonal to q

        :return: squared distance between projected and dispered angles
        """

        N, d = X.shape

        device = X.device

        p, q = init_great_circle(d, dtype=X.dtype, device=device)

        Xp = X @ p
        Xq = X @ q

        return _dist_along_slice(Xp, Xq, N, device)


class AxisAlignedSlicedSphereDispersion(SphereDispersion):

    def forward(self, X: Tensor) -> Tensor:
        """
        calculates forward pass for sliced dispersion
        :param reduction:
        :param return_hidden_states: return values to calculate grad
        :param X: Tensor of shape (N, d)
        :param p, q: vectors defining great circle. p is orthogonal to q

        :return: squared distance between projected and dispered angles
        """

        N, d = X.shape


        device = X.device


        i, j = torch.randperm(d)[:2]

        Xp = X[:, i]
        Xq = X[:, j]

        return _dist_along_slice(Xp, Xq, N, device)
