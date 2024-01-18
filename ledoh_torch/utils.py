import torch


def minimum_acos_distance(X: torch.Tensor) -> torch.Tensor:
    dist = torch.acos((X @ X.T).clamp(-1+1e-4, 1-1e-4))
    dist.fill_diagonal_(float('inf'))
    return dist.view(-1).min()


def circular_variance(X: torch.Tensor) -> torch.Tensor:
    return 1 - torch.norm(torch.mean(X, dim=0))


def init_great_circle(d, dtype=None, device=None):
    PQ = torch.randn(d, 2,
                     dtype=dtype,
                     device=device)

    PQ, R = torch.linalg.qr(PQ)
    PQ *= R.diagonal().sign()
    p, q = PQ[:, 0], PQ[:, 1]
    return p, q
