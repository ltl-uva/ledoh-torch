from typing import Tuple
import torch
@torch.jit.script
def acos_distance_matrix(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    return torch.acos((X @ Y.T).clamp(-1 + 1e-9, 1 - 1e-9))
@torch.jit.script
def minimum_acos_distance(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    adist = acos_distance_matrix(X,Y)
    return adist[adist>0].min()

@torch.jit.script
def minimum_acos_distance_row(X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    dists = acos_distance_matrix(X,Y)
    return dists.sort(dim=1).values[:,1]

@torch.jit.script
def minimum_acos_distance_block(X: torch.Tensor, block_size:int = 1024) -> torch.Tensor:
    X_split = torch.split(X, block_size, dim=0)
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
