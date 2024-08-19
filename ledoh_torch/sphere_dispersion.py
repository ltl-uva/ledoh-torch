import torch
from torch import Tensor


class SphereDispersion(torch.nn.Module):

    def forward(self, *args, **kwargs) -> Tensor:
        """
        :param args: positional arguments
        :param kwargs: keyword arguments, must include reduction method; sum or mean
        :return: First return parameter is loss value, second is dict with any extra information.
        Dict MUST contain sample size
        """
        return torch.tensor(0)
