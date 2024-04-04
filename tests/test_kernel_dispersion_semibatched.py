from ledoh_torch import KernelSphereSemibatchDispersion
from test_utils import run_dummy_training_loop
import torch
import torch.nn.functional as F


def basic_test(d):
    X = F.normalize(10 + torch.randn(1000, d, requires_grad=True), dim=-1)
    run_dummy_training_loop(X, KernelSphereSemibatchDispersion.forward, reg_params={"gamma": 1}, n_steps=1000, lr=0.005, d=16)



if __name__ == "__main__":
    basic_test(d=16)


