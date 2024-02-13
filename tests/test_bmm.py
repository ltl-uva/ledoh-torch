from typing import Tuple
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

@torch.jit.script
def minimum_acos_distance(X: torch.Tensor) -> torch.Tensor:
    return torch.acos((X @ X.T).clamp(-1 + 1e-4, 1 - 1e-4)).fill_diagonal_(float('inf')).view(-1).min()


def minimum_acos_distance_block(X: torch.Tensor, block_size:int = 16) -> torch.Tensor:
    @torch.jit.script
    def _multiply_block(A:torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.acos((A @ B.T).clamp(-1 + 1e-4, 1 - 1e-4)).fill_diagonal_(float('inf')).view(-1).min()

    X_split = torch.split(X, block_size, dim=0)
    mins = []
    for A in X_split:
        for B in X_split:
            mins.append(_multiply_block(A, B))
    return torch.stack(mins).min()

if __name__=="__main__":
    X = F.normalize(torch.randn(50000, 128), dim=-1)
    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True,record_shapes=True) as prof:
        with record_function("acos_dist"):
            a = minimum_acos_distance(X)
        with record_function("acos_dist_block"):
            b = minimum_acos_distance_block(X, 128)

    print(a,b, torch.allclose(a,b))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))