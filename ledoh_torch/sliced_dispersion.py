import math
from typing import Optional

import torch
from torch import Tensor

from .utils import init_great_circle
from .sphere_dispersion import SphereDispersion


class SlicedSphereDispersion(SphereDispersion):
    @staticmethod
    def forward(X: Tensor,
                p: Optional[Tensor],
                q: Optional[Tensor],
                ) -> Tensor:
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

        if p is None or q is None:
            p, q = init_great_circle(d, dtype=X.dtype, device=device)

        Xp = X @ p
        Xq = X @ q

        thetas = torch.arctan2(Xq, Xp)

        ix = torch.argsort(thetas)
        invix = torch.empty_like(ix)
        invix[ix] = torch.arange(N, device=device)

        phis = 2 * math.pi * torch.arange(1, N+1, device=device) / N
        phis -= math.pi + math.pi / N  # make zero-mean
        phis = phis[invix]

        thetas_star = torch.mean(thetas) + phis

        dist = .5 * torch.sum(torch.pow(thetas - thetas_star, 2))

        return dist
