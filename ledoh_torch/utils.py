from typing import Tuple
import torch

@torch.jit.script
def get_acos_distance_matrix(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    return torch.acos((X @ Y.T).clamp(-1 + 1e-7, 1 - 1e-7))

@torch.jit.script
def minimum_acos_distance(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    return get_acos_distance_matrix(X,Y).fill_diagonal_(torch.inf).min()

@torch.jit.script
def avg_acos_distance(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    distances = get_acos_distance_matrix(X, Y)

    return 2*(distances-torch.tril(distances)).mean()

@torch.jit.script
def avg_acos_distance_batch(X: torch.Tensor, batch_size: int=1024) -> torch.Tensor:
    X_split = torch.split(X, batch_size, dim=0)
    avgs = []
    for A in X_split:
        for B in X_split:
            avgs.append(avg_acos_distance(A, B))
    return torch.stack(avgs).mean()

@torch.jit.script
def median_acos_distance(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    distances = get_acos_distance_matrix(X, Y)
    distances -=torch.tril(distances)
    distances = distances[distances>0]
    return distances.median()

@torch.jit.script
def median_acos_distance_batch(X: torch.Tensor, batch_size: int=1024) -> torch.Tensor:
    X_split = torch.split(X, batch_size, dim=0)
    medians = []
    for A in X_split:
        for B in X_split:
            medians.append(median_acos_distance(A, B))
    return torch.stack(medians).median()

@torch.jit.script
def minimum_acos_distance_row(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    return get_acos_distance_matrix(X,Y).fill_diagonal_(torch.inf).min(dim=1).values


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
