from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .sphere_dispersion import SphereDispersion


class LloydSphereDispersion(SphereDispersion):
    def __init__(self, n_samples: int=100):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, X: Tensor) -> Tensor:
        """Compute the dispersion of a set of points on the sphere using Lloyd's algorithm.
        :param reduction:
        :param X: points on the sphere
        :param n_samples:
        :return: mean min squared distance between samples and nearest points on the sphere, {"sample_size": 1}
        """
        d = X.size(-1)
        device = X.device
        samples = F.normalize(torch.randn(self.n_samples,1, d,device=device), dim=-1)
        # compute the distance to the nearest point on the sphere
        dist2 = torch.acos((samples @ X.T).clamp(-1+1e-7, 1-1e-7)) ** 2
        # choose the closest center for each sample and sum results
        #since we compute mean directly, return sample_size=1
        loss = (torch.min(dist2, dim=-1)[0]).mean(dim=0)

        return loss
