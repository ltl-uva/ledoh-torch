from ledoh_torch import KernelSphereDispersion
from ledoh_torch import minimum_cosine_distance, circular_variance
import torch
import torch.nn.functional as F
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import SphereExact
from geoopt.optim import RiemannianAdam

if __name__ == "__main__":
    n_steps = 1000
    manifold = SphereExact()
    X = F.normalize(torch.randn(1000, 16, requires_grad=True), dim=-1)
    X = ManifoldParameter(X, manifold=manifold)
    optimizer = RiemannianAdam([X], stabilize=1, lr=0.005)
    losses = []
    min_dists = [minimum_cosine_distance(X).item()]
    circ_vars = [circular_variance(X).item()]

    for i in range(n_steps):
        optimizer.zero_grad()
        loss, _ = KernelSphereDispersion.forward(X, gamma=1)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        circ_vars.append(circular_variance(X).item())
        min_dists.append(minimum_cosine_distance(X).item())

    print("Dispersion: ", losses[0], losses[-1])
    print("Min dist: ", min_dists[0], min_dists[-1])
    print("Circ var: ", circ_vars[0], circ_vars[-1])

