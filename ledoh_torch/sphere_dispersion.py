from typing import Tuple, Any, Dict

import torch
from torch import Tensor


class SphereDispersion:
    @staticmethod
    def forward(X, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """
        :param X: points on the sphere that have to be dispersed
        :param kwargs: any additional parameters
        :return: First return parameter is loss value, second is dict with any extra information.
        Dict MUST contain sample size
        """
        return torch.tensor(0), {"sample_size": 0}
