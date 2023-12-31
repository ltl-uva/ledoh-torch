import torch
import torch.nn.functional as F

from ledoh_torch import LloydSphereDispersion
from test_utils import run_dummy_training_loop

if __name__ == "__main__":
        d = 16
        X = F.normalize(torch.randn(1000, d, requires_grad=True), dim=-1)
        run_dummy_training_loop(X, LloydSphereDispersion.forward, reg_params={"n_clusters": 2}, n_steps=1000, lr=0.005, d=16)
