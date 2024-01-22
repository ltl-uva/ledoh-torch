from ledoh_torch import init_great_circle, compute_sliced_sphere_dispersion
from ledoh_torch import minimum_acos_distance, circular_variance
import torch
import torch.nn.functional as F
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Sphere
from geoopt.optim import RiemannianAdam
import matplotlib.pyplot as plt

from test_utils import run_dummy_training_loop

def test_with_same_great_circle():
    d = 16
    p, q = init_great_circle(d)
    X = F.normalize(torch.randn(1000, d, requires_grad=True), dim=-1)
    run_dummy_training_loop(X, compute_sliced_sphere_dispersion, reg_params={"p": p, "q": q}, n_steps=1000, lr=0.005, d=d)

def test_with_different_great_circle():
    """
    duplicate code cause I am lazy
    """
    d = 16
    n_steps = 1000
    manifold = SphereExact()
    X = F.normalize(torch.randn(1000, d, requires_grad=True), dim=-1)
    X = ManifoldParameter(X, manifold=manifold)
    optimizer = RiemannianAdam([X], stabilize=1, lr=0.005)
    losses = []
    min_dists = [minimum_acos_distance(X).item()]
    circ_vars = [circular_variance(X).item()]

    for i in range(n_steps):
        optimizer.zero_grad()
        p, q = init_great_circle(d)
        loss, _ = compute_sliced_sphere_dispersion(X, p,q)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        circ_vars.append(circular_variance(X).item())
        min_dists.append(minimum_acos_distance(X).item())

    print("Dispersion: ", losses[0], losses[-1])
    print("Min dist: ", min_dists[0], min_dists[-1])
    print("Circ var: ", circ_vars[0], circ_vars[-1])

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

def project_and_plot(X, p, q, color='b', alpha=0.5):
    fig, ax = plt.subplots()
    Xp = X @ p
    Xq = X @ q

    thetas = torch.arctan2(Xq, Xp)
    x, y = points_on_circle(thetas)

    ax.scatter(x, y, color=color, alpha=alpha)
    plt.show()

def test_and_plot():
    d = 3

    X = F.normalize(torch.randn(100, d, requires_grad=True), dim=-1)

    n_steps = 10
    manifold = Sphere()
    X = ManifoldParameter(X, manifold=manifold)

    optimizer = RiemannianAdam([X], stabilize=1, lr=0.005)
    losses = []
    min_dists = [minimum_acos_distance(X.detach()).item()]
    circ_vars = [circular_variance(X.detach()).item()]

    for i in range(n_steps):
        p, q = init_great_circle(d)
        project_and_plot(X.detach(), p, q)
        optimizer.zero_grad()
        loss = compute_sliced_sphere_dispersion(X,p,q)
        losses.append(loss.detach().item())

        loss.backward()
        optimizer.step()

        project_and_plot(X.detach(), p, q, color='r', alpha=0.5)
        circ_vars.append(circular_variance(X.detach()).item())
        min_dists.append(minimum_acos_distance(X.detach()).item())

    print("Dispersion: ", losses[0], losses[-1])
    print("Min dist: ", min_dists[0], min_dists[-1])
    print("Circ var: ", circ_vars[0], circ_vars[-1])

if __name__ == "__main__":
    test_and_plot()