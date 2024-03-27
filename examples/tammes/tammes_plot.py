import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Sphere
from geoopt.optim import RiemannianSGD, RiemannianAdam

from ledoh_torch import (
    SlicedSphereDispersion,
    circular_variance,
    KernelSphereDispersion,
    LloydSphereDispersion,
    minimum_acos_distance_row
)



def train_tammes(n, d, target, epochs, lr, loss_fn, loss_params, verbose=False):
    torch.manual_seed(42)
    X_init = F.normalize(torch.randn(n, d, dtype=torch.float32), dim=-1)
    manifold = Sphere()
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    X_start = X.detach().clone()

    opt = RiemannianAdam([X], lr=lr, stabilize=1)
    losses = []

    for i in range(epochs):
        opt.zero_grad()
        loss = loss_fn.forward(X, **loss_params)
        losses.append(loss.detach().item())
        loss.backward()
        opt.step()

    start_angles = torch.rad2deg(minimum_acos_distance_row(X_start,X_start))
    end_angles = torch.rad2deg(minimum_acos_distance_row(X.detach(),X.detach()))

    if verbose:
        print(f"Angle")
        print(f"\tMin.: {end_angles.min()}")
        print(f"\tTarget: {target}")
        print(f"Variance")
        print(f"\tStart: {circular_variance(X_start)}")
        print(f"\tEnd: {circular_variance(X.detach())}")
        print(f"Error: {100 - (end_angles.min() / target * 100):.2f}%")

    return start_angles, end_angles


def plot_angles_(angles, ax, target=None, target_color=None, *args, **kwargs):
    " target is desired min angle "
    if target is not None:
        ax.plot([target] * len(angles), label="Optimal min. angle", color=target_color)

    angles = sorted(angles)

    if "marker" not in kwargs:
        kwargs["marker"] = "."

    ax.plot(angles, linestyle='', *args, **kwargs)
    ax.set_xticks([], [])


def main(out_file=None):
    epochs = 2500

    # n = 13
    tammes_n13 = (13, 3, 57.1367031)
    n, d, target = tammes_n13
    print(tammes_n13)

    mmd_angles_n13 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.05,
        loss_fn=KernelSphereDispersion,
        loss_params=dict(gamma=20, batch_size=-1),
        verbose=True
    )
    sliced_angles_n13 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.005,
        loss_fn=SlicedSphereDispersion,
        loss_params=dict(p=None, q=None),
        verbose=True
    )
    lloyd_angles_n13 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.005,
        loss_fn= LloydSphereDispersion,
        loss_params=dict(n_samples=6000),
        verbose=True
    )

    # n = 14
    tammes_n14 = (14, 3, 55.6705700)
    n, d, target = tammes_n14
    print(tammes_n14)

    mmd_angles_n14 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.05,
        loss_fn=KernelSphereDispersion,
        loss_params=dict(gamma=20, batch_size=-1),
        verbose=True
    )
    sliced_angles_n14 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.005,
        loss_fn=SlicedSphereDispersion,
        loss_params=dict(p=None, q=None),
        verbose=True
    )
    lloyd_angles_n14 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.005,
        loss_fn= LloydSphereDispersion,
        loss_params=dict(n_samples=8000),
        verbose=True
    )

    # n = 30
    tammes_n30 = (30, 3, 38.6)
    n, d, target = tammes_n30
    print(tammes_n30)

    mmd_angles_n30 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.05,
        loss_fn=KernelSphereDispersion,
        loss_params=dict(gamma=45, batch_size=-1),
        verbose=True
    )
    sliced_angles_n30 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.005,
        loss_fn=SlicedSphereDispersion,
        loss_params=dict(p=None, q=None),
        verbose=True
    )
    lloyd_angles_n30 = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=0.005,
        loss_fn= LloydSphereDispersion,
        loss_params=dict(n_samples=10000),
        verbose=True
    )

    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=15)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21,5), sharey=True)

    # n = 13
    ax_idx = 0
    plot_angles_(mmd_angles_n13[1], ax=axes[ax_idx], target=tammes_n13[-1], target_color="black", label="MMD", color="blue", marker="o")
    plot_angles_(sliced_angles_n13[1], ax=axes[ax_idx], label="Sliced", color="green", marker="s")
    plot_angles_(lloyd_angles_n13[1], ax=axes[ax_idx], label="Lloyd", color="red", marker="^")
    axes[ax_idx].title.set_text(f"N = {tammes_n13[0]}")
    axes[ax_idx].set_ylim([0, 90])
    axes[ax_idx].set_ylabel("Minimum angle")


    # n = 14
    ax_idx = 1
    plot_angles_(mmd_angles_n14[1], ax=axes[ax_idx], target=tammes_n14[-1], target_color="black", label="MMD", color="blue", marker="o")
    plot_angles_(sliced_angles_n14[1], ax=axes[ax_idx], label="Sliced", color="green", marker="s")
    plot_angles_(lloyd_angles_n14[1], ax=axes[ax_idx], label="Lloyd", color="red", marker="^")
    axes[ax_idx].title.set_text(f"N = {tammes_n14[0]}")

    # n = 30
    ax_idx = 2
    plot_angles_(mmd_angles_n30[1], ax=axes[ax_idx], target=tammes_n30[-1], target_color="black", label="MMD", color="blue", marker="o")
    plot_angles_(sliced_angles_n30[1], ax=axes[ax_idx], label="Sliced", color="green", marker="s")
    plot_angles_(lloyd_angles_n30[1], ax=axes[ax_idx], label="Lloyd", color="red", marker="^")
    axes[ax_idx].title.set_text(f"N = {tammes_n30[0]}")
    axes[ax_idx].legend()


    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    main(out_file="tammes.pdf")
