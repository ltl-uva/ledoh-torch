from typing import Tuple
import torch

@torch.jit.script
def minimum_acos_distance(X: torch.Tensor) -> torch.Tensor:
    return torch.acos((X @ X.T).clamp(-1 + 1e-4, 1 - 1e-4)).fill_diagonal_(float('inf')).view(-1).min()


def minimum_acos_distance_block(X: torch.Tensor, block_size:int = 1024) -> torch.Tensor:
    @torch.jit.script
    def _multiply_block(A:torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.acos((A @ B.T).clamp(-1 + 1e-4, 1 - 1e-4)).fill_diagonal_(float('inf')).view(-1).min()

    X_split = torch.split(X, block_size, dim=0)
    mins = []
    for A in X_split:
        for B in X_split:
            mins.append(_multiply_block(A, B))
    return torch.stack(mins).min()


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
