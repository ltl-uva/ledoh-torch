from typing import Tuple, Any, Dict

import torch
from torch import Tensor


class SphereDispersion:
    @staticmethod
    def forward(*args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """
        :param args: positional arguments
        :param kwargs: keyword arguments, must include reduction method; sum or mean
        :return: First return parameter is loss value, second is dict with any extra information.
        Dict MUST contain sample size
        """
        return torch.tensor(0), {"sample_size": 0}
