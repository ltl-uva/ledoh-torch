import math
import torch
import torch.nn.functional as F

from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianSGD
from geoopt.manifolds import SphereExact

import matplotlib.pyplot as plt

from ledoh_torch import init_great_circle, SlicedSphereDispersion


def project_and_plot(X, p, q, ax, color='b', alpha=0.5):
    Xp = X @ p
    Xq = X @ q
    thetas = torch.arctan2(Xq, Xp)
    _plot(thetas, ax, color, alpha)


def _plot(thetas, ax, color='b', alpha=0.5):
    x = torch.cos(thetas)
    y = torch.sin(thetas)
    ax.scatter(x, y, color=color, alpha=alpha)


def distance(X, p, q):
    N = X.size(0)
    device = X.device

    Xp = X @ p
    Xq = X @ q

    thetas = torch.arctan2(Xq, Xp)
    ix = torch.argsort(thetas)
    invix = torch.empty_like(ix)
    invix[ix] = torch.arange(N, device=device)

    phis = 2 * math.pi * torch.arange(1, N+1, device=device) / N
    phis = -math.pi - math.pi / N + phis
    phis = phis[invix]

    thetas_star = torch.mean(thetas) + phis
    dist = 0.5 * torch.sum(torch.pow(thetas - thetas_star, 2))
    return dist, thetas_star


def main():
    d = 10

    torch.manual_seed(40)

    n_points = 5
    X_init = F.normalize(10 + torch.randn(n_points, d), dim=-1)

    manifold = SphereExact()
    p, q = init_great_circle(d)


    fig, axes = plt.subplots(nrows=1, ncols=4)
    for ax in axes:
        ax.add_patch(plt.Circle([0,0], 1, fill=False))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

    project_and_plot(X_init, p, q, axes[0])

    # round zero
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    dist, thetas_star = distance(X, p, q)
    _plot(thetas_star.detach(), axes[1])

    # round one
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    optimizer = RiemannianSGD([X], stabilize=1, lr=1)
    optimizer.zero_grad()
    loss, _ = distance(X, p, q)
    loss.backward()
    grad = X.grad
    print(grad)
    # check this gradient is orthogonal to X itself.
    # print((X * grad).sum(dim=-1))
    # update X with riemannian gradient
    X = manifold.expmap(X, -grad)
    optimizer.step()
    project_and_plot(X.detach(), p, q, axes[2])

    # round two
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    optimizer = RiemannianSGD([X], stabilize=1, lr=1)
    optimizer.zero_grad()
    loss = SlicedSphereDispersion.forward(X, p, q)
    loss.backward()
    optimizer.step()
    project_and_plot(X.detach(), p, q, axes[3])
    axes[0].set_title("before update")
    axes[1].set_title("thetas_star")
    axes[2].set_title("after update -- autograd")
    axes[3].set_title("after update -- new impl")
    plt.show()


if __name__ == "__main__":
    main()
