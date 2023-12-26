import torch
from torch import Tensor

def cosine_kernel_dispersion(X: Tensor, gamma: float, sample_size=-1):
    if sample_size<0:
        sample_size=X.shape[0]
    batch_idx=torch.randperm(X.shape[0], device=X.device)[:sample_size]
    X_batch = torch.index_select(X, 0, batch_idx)
    print(X_batch.shape)
    similarities = X_batch @ X_batch.T
    similarities.fill_diagonal_(0)
    return torch.exp(gamma*similarities).sum()