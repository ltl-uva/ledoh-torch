import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Sphere
from geoopt.optim import RiemannianAdam

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


def add_dispersed_points_to_ax(initial_angles, mmd_angles, lloyd_angles, sliced_angles, tammes_config, ax):
    n, _, target = tammes_config
    ax.plot([target] * len(mmd_angles), label="Optimal min. angle", color="black")
    plot_angles_(initial_angles, ax=ax, label="Initial", color="black")
    plot_angles_(mmd_angles, ax=ax, label="MMD", color="blue", marker="o")
    plot_angles_(sliced_angles, ax=ax, label="Sliced", color="green", marker="s")
    plot_angles_(lloyd_angles, ax=ax, label="Lloyd", color="red", marker="^")
    ax.title.set_text(f"N = {n}")
    ax.set_ylim([0, 90])


def _run(tammes_config, epochs, lrs, mmd_params, lloyd_params, sliced_params, verbose):
    n, d, target = tammes_config
    print(tammes_config)

    initial_angles, mmd_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["mmd"],
        loss_fn=KernelSphereDispersion,
        loss_params=mmd_params,
        verbose=verbose
    )
    _, sliced_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["sliced"],
        loss_fn=SlicedSphereDispersion,
        loss_params=sliced_params,
        verbose=verbose
    )
    _, lloyd_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["lloyd"],
        loss_fn= LloydSphereDispersion,
        loss_params=lloyd_params,
        verbose=verbose
    )

    return initial_angles, mmd_angles, sliced_angles, lloyd_angles


def main(out_file=None, verbose=False):
    epochs = 2500
    lrs = dict(mmd=0.05, sliced=0.005, lloyd=0.005)

    # n = 13
    tammes_n13 = (13, 3, 57.1367031)
    initial_angles_n13, mmd_angles_n13, sliced_angles_n13, lloyd_angles_n13 = _run(
        tammes_config=tammes_n13, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=20, batch_size=-1), lloyd_params=dict(n_samples=200),
        sliced_params=dict(p=None, q=None), verbose=verbose
    )

    # n = 14
    tammes_n14 = (14, 3, 55.6705700)
    initial_angles_n14, mmd_angles_n14, sliced_angles_n14, lloyd_angles_n14 = _run(
        tammes_config=tammes_n14, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=20, batch_size=-1), lloyd_params=dict(n_samples=200),
        sliced_params=dict(p=None, q=None), verbose=verbose
    )

    # n = 30
    tammes_n30 = (30, 3, 38.6)
    initial_angles_n30, mmd_angles_n30, sliced_angles_n30, lloyd_angles_n30 = _run(
        tammes_config=tammes_n30, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=45, batch_size=-1), lloyd_params=dict(n_samples=300),
        sliced_params=dict(p=None, q=None), verbose=verbose
    )

    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=15)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21,5), sharey=True)

    # plot
    add_dispersed_points_to_ax(
        initial_angles=initial_angles_n13, mmd_angles=mmd_angles_n13,
        lloyd_angles=lloyd_angles_n13, sliced_angles=sliced_angles_n13,
        tammes_config=tammes_n13, ax=axes[0]
    )
    add_dispersed_points_to_ax(
        initial_angles=initial_angles_n14, mmd_angles=mmd_angles_n14,
        lloyd_angles=lloyd_angles_n14, sliced_angles=sliced_angles_n14,
        tammes_config=tammes_n14, ax=axes[1])
    add_dispersed_points_to_ax(
        initial_angles=initial_angles_n30, mmd_angles=mmd_angles_n30,
        lloyd_angles=lloyd_angles_n30, sliced_angles=sliced_angles_n30,
        tammes_config=tammes_n30, ax=axes[2]
    )
    axes[0].set_ylabel("Minimum angle")
    axes[2].legend()

    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    main(out_file="tammes.pdf")
