import torch

def minimum_cosine_distance(X: torch.Tensor) -> torch.Tensor:
    dp = torch.einsum('ij,kj->ik ', X, X)
    dist = torch.acos(dp)
    return torch.triu(dist).min()


def circular_variance(X: torch.Tensor) -> torch.Tensor:
    return 1 - torch.norm(torch.mean(X, dim=0))

def init_great_circle(d, dtype=None, device=None):
    PQ = torch.randn(d, 2,
                     dtype=dtype,
                     device=device)

    PQ, R = torch.linalg.qr(PQ)
    PQ *= R.diagonal().sign()
    p, q = PQ[:, 0], PQ[:, 1]
    return p,q