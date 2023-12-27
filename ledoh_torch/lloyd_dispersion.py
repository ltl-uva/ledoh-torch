from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .sphere_dispersion import SphereDispersion


class LloydSphereDispersion(SphereDispersion):
    @staticmethod
    def forward(X, reduction='mean', n_samples=100) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute the dispersion of a set of points on the sphere using Lloyd's algorithm.
        :param reduction:
        :param X: points on the sphere
        :param n_samples:
        :return: mean min squared distance between samples and nearest points on the sphere, {"sample_size": 1}
        """
        d = X.size(-1)
        device = X.device
        samples = F.normalize(torch.randn(n_samples,
                                          1,
                                          d,
                                          device=device
                                          ),
                              dim=-1)
        # compute the distance to the nearest point on the sphere
        dist2 = torch.acos(samples @ X.T) ** 2
        # choose the closest center for each sample and sum results
        #since we compute mean directly, return sample_size=1
        loss = (torch.min(dist2, dim=-1)[0]).sum()
        if reduction == "mean":
            loss = torch.div(loss, n_samples)
            return loss, {"sample_size": 1}
        else:
            return loss, {"sample_size": n_samples}
