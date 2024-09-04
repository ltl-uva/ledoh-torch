from typing import Tuple
import torch

@torch.jit.script
def get_acos_distance_matrix(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    mask = (1 - torch.eye(X.shape[0], Y.shape[0])).to(dtype=torch.bool, device=X.device)
    return torch.acos((X @ Y.T)[mask].clamp(-1 + 1e-6, 1 - 1e-6))

@torch.jit.script
def minimum_acos_distance(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    return get_acos_distance_matrix(X,Y).min()

@torch.jit.script
def avg_acos_distance(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    return get_acos_distance_matrix(X,Y).mean()

@torch.jit.script
def avg_acos_distance_batch(X: torch.Tensor, batch_size=1024) -> torch.Tensor:
    X_split = torch.split(X, batch_size, dim=0)
    avgs = []
    for A in X_split:
        for B in X_split:
            avgs.append(avg_acos_distance(A, B))
    return torch.stack(avgs).mean()

@torch.jit.script
def minimum_acos_distance_row(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    return get_acos_distance_matrix(X,Y).min(dim=1).values


@torch.jit.script
def minimum_acos_distance_batch(X: torch.Tensor, batch_size:int = 1024) -> torch.Tensor:
    X_split = torch.split(X, batch_size, dim=0)
    mins = []
    for A in X_split:
        for B in X_split:
            mins.append(minimum_acos_distance(A, B))
    return torch.stack(mins).min()

@torch.jit.script
def circular_variance(X: torch.Tensor) -> torch.Tensor:
    return 1 - torch.norm(torch.mean(X, dim=0))

def init_great_circle(d: int, dtype=None, device: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    PQ = torch.randn(d, 2,
                     dtype=dtype,
                     device=device)

    PQ, R = torch.linalg.qr(PQ)
    PQ *= R.diagonal().sign()
    p, q = PQ[:, 0], PQ[:, 1]
    return p, q
