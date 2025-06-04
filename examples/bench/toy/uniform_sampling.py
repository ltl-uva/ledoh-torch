import torch

from ledoh_torch import get_acos_distance_matrix

n_runs=10000
ns = [1000,10000]
dims = [64,512,4096]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with torch.no_grad:
    for n in ns:
        for d in dims:
            runs_angles = []
            for _ in range(n_runs):
               points = torch.randn(n, d, dtype=torch.float32, device=device)
               points = points / torch.norm(points, dim=-1, keepdim=True)

               angles = get_acos_distance_matrix(points, points)
               angles = angles.fill_diagonal_(torch.inf)
               runs_angles.append(angles.min(dim=1)[0].tolist())
            runs_angles = torch.tensor(runs_angles, dtype=torch.float32, device=device)
            mean_angles = runs_angles.mean(dim=0)
            std_angles = runs_angles.std(dim=0)
            print(f"n={n}, d={d}, mean min angle: {mean_angles.mean().item():.4f}, std: {std_angles.mean().item():.4f}")

