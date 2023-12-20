import torch
from torch import Tensor

def mmd_dispersion(X: Tensor, gamma: float):
    similarities = X @ X.T
    similarities.fill_diagonal_(0)
    return torch.exp(gamma*similarities).sum()