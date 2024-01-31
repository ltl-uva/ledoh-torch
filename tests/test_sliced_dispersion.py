from ledoh_torch import init_great_circle, compute_sliced_sphere_dispersion
from ledoh_torch import minimum_acos_distance, circular_variance
import torch
import torch.nn.functional as F
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import SphereExact
from geoopt.optim import RiemannianAdam
import matplotlib.pyplot as plt

from test_utils import run_dummy_training_loop

def points_on_circle(angle):
    '''
        Finding the x,y coordinates on circle, based on given angle
    '''
    from math import cos, sin, pi
    #center of circle, angle in degree and radius of circle
    center = [0.5,0.5]
    radius = 1
    x = [center[0] + (radius * cos(a)) for a in angle]
    y = [center[1] + (radius * sin(a)) for a in angle]

    return x,y

def project_and_plot(X, p, q, ax, color='b', alpha=0.5):

    Xp = X @ p
    Xq = X @ q

    thetas = torch.arctan2(Xq, Xp)
    x, y = points_on_circle(thetas)

    ax.scatter(x, y, color=color, alpha=alpha)
    #plt.show()

def test_and_plot():
    d = 3

    X = F.normalize(torch.randn(3, d, requires_grad=True), dim=-1)

    n_steps = 1
    manifold = SphereExact()
    X = ManifoldParameter(X, manifold=manifold)

    optimizer = RiemannianAdam([X], stabilize=1, lr=1)
    losses = []
    min_dists = [minimum_acos_distance(X.detach()).item()]
    circ_vars = [circular_variance(X.detach()).item()]
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10,10))

    for i in range(n_steps):
        p, q = init_great_circle(d)
        project_and_plot(X.detach(), p, q, ax[i % 4][i // 4], color='r', alpha=0.5)
        optimizer.zero_grad()
        loss = compute_sliced_sphere_dispersion(X,p,q)
        losses.append(loss.detach().item())

        loss.backward()
        optimizer.step()
        project_and_plot(X.detach(), p, q, ax[i % 4][i // 4], color='b', alpha=0.5)
        circ_vars.append(circular_variance(X.detach()).item())
        min_dists.append(minimum_acos_distance(X.detach()).item())

    print("Dispersion: ", losses[0], losses[-1])
    print("Min dist: ", min_dists[0], min_dists[-1])
    print("Circ var: ", circ_vars[0], circ_vars[-1])
    plt.show()

if __name__ == "__main__":
    test_and_plot()