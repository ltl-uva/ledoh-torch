from ledoh_torch import minimum_acos_distance, circular_variance
import torch
import torch.nn.functional as F
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import SphereExact
from geoopt.optim import RiemannianAdam


def run_dummy_training_loop(X, regularizer_function, reg_params=None, n_steps=1000, lr=0.005, d=16):
    n_steps = n_steps
    manifold = SphereExact()
    X = ManifoldParameter(X, manifold=manifold)
    optimizer = RiemannianAdam([X], stabilize=1, lr=lr)
    losses = []
    min_dists = [minimum_acos_distance(X).item()]
    circ_vars = [circular_variance(X).item()]

    for i in range(n_steps):
        optimizer.zero_grad()
        loss, _ = regularizer_function(X, *reg_params)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        circ_vars.append(circular_variance(X).item())
        min_dists.append(minimum_acos_distance(X).item())

    print("Dispersion: ", losses[0], losses[-1])
    print("Min dist: ", min_dists[0], min_dists[-1])
    print("Circ var: ", circ_vars[0], circ_vars[-1])