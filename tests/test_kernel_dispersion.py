from ledoh_torch import KernelSphereDispersion
from test_utils import run_dummy_training_loop
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    d = 16
    X = F.normalize(torch.randn(1000, d, requires_grad=True), dim=-1)
    run_dummy_training_loop(X, KernelSphereDispersion.forward, reg_params={"gamma": 1}, n_steps=1000, lr=0.005, d=16)


