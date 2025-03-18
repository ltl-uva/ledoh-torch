import random

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import SphereExact
from geoopt.optim import RiemannianAdam

from ledoh_torch import (
    SlicedSphereDispersion,
    circular_variance,
    KernelSphereDispersion,
    LloydSphereDispersion,
    MMADispersion,
    MHEDispersion,
    KoLeoDispersion,
    AxisAlignedBatchSphereDispersion,
    minimum_acos_distance_row
)

from matplotlib import colors

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def train_tammes(n, d, target, epochs, lr, loss_fn, verbose=False):
    torch.manual_seed(42)
    X_init = F.normalize(torch.randn(n, d, dtype=torch.float32), dim=-1)
    manifold = SphereExact()
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    X_start = X.detach().clone()

    opt = RiemannianAdam([X], lr=lr, stabilize=1)
    losses = []

    for i in range(epochs):
        opt.zero_grad()
        loss = loss_fn(X)
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

def plot_angles_violin_(angles, ax, target=None, target_color=None, *args, **kwargs):
    ax.violinplot(angles)


def add_dispersed_points_to_ax(initial_angles, mmd_angles, lloyd_angles, sliced_angles, tammes_config, ax):
    n, _, target = tammes_config
    ax.plot([target] * len(mmd_angles), label="Optimal min. angle", color="black")
    plot_angles_(initial_angles, ax=ax, label="Initial", color="black")
    plot_angles_(mmd_angles, ax=ax, label="MMD", color="blue", marker="o")
    plot_angles_(sliced_angles, ax=ax, label="Sliced", color="green", marker="s")
    plot_angles_(lloyd_angles, ax=ax, label="Lloyd", color="red", marker="^")
    ax.title.set_text(f"N = {n}")
    ax.set_ylim([0, 90])


def _run(tammes_config, epochs, lrs, mmd_params, lloyd_params, sliced_params, mma_params, mhe_params, verbose):
    n, d, target = tammes_config
    print(tammes_config)

    initial_angles, mmd_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["mmd"],
        loss_fn=KernelSphereDispersion(**mmd_params),
        verbose=verbose
    )
    _, sliced_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=5000,
        lr=lrs["sliced"],
        loss_fn=SlicedSphereDispersion(),
        verbose=verbose
    )
    _, lloyd_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["lloyd"],
        loss_fn= LloydSphereDispersion(**lloyd_params),
        verbose=verbose
    )

    _, mma_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["mma"],
        loss_fn=MMADispersion(**mma_params),
        verbose=verbose
    )
    _, mhe_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["mhe"],
        loss_fn=MHEDispersion(**mhe_params),
        verbose=verbose
    )
    _, koleo_angles = train_tammes(
        n=n,
        d=d,
        target=target,
        epochs=epochs,
        lr=lrs["koleo"],
        loss_fn=KoLeoDispersion(),
        verbose=verbose
    )

    return initial_angles, mmd_angles, sliced_angles, lloyd_angles, mma_angles, mhe_angles, koleo_angles


def main(out_file=None, verbose=False):
    epochs = 2500
    lrs = dict(mmd=0.005, sliced=0.005, lloyd=0.005,mma=0.005, mhe=0.005, koleo=0.005)

    n = 13
    tammes_n13 = (13, 3, 57.1367031)
    initial_angles_n13, mmd_angles_n13,sliced_angles_n13, lloyd_angles_n13, \
    mma_angles_n13, mhe_angles_n13, koleo_angles_n13, = _run(
        tammes_config=tammes_n13, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=20, batch_size=-1), lloyd_params=dict(n_samples=200),
        sliced_params=dict(), mma_params=dict(batch_size=-1),
        mhe_params=dict(batch_size=-1), verbose=verbose
    )

    # n = 14
    tammes_n14 = (14, 3, 55.6705700)
    initial_angles_n14, mmd_angles_n14, sliced_angles_n14, lloyd_angles_n14, \
        mma_angles_n14, mhe_angles_n14, koleo_angles_14= _run(
        tammes_config=tammes_n14, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=20, batch_size=-1), lloyd_params=dict(n_samples=200),
        sliced_params=dict(), mma_params=dict(batch_size=-1),
        mhe_params=dict(batch_size=-1), verbose=verbose
    )

    n = 24
    tammes_n24 = (24, 3, 48.53529763)
    initial_angles_n24, mmd_angles_n24, \
    sliced_angles_n24, lloyd_angles_n24, \
    mma_angles_n24, mhe_angles_n24, koleo_angles_n24 = _run(
        tammes_config=tammes_n24, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=25, batch_size=-1), lloyd_params=dict(n_samples=300),
        sliced_params=dict(), mma_params=dict(batch_size=-1),
        mhe_params=dict(batch_size=-1), verbose=verbose
    )

    n = 128
    tammes_n128 = (128, 3, 18.6349726)
    initial_angles_n128, mmd_angles_n128, \
    sliced_angles_n128, lloyd_angles_n128, \
    mma_angles_n128, mhe_angles_n128, koleo_angles_n128 = _run(
        tammes_config=tammes_n128, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=25, batch_size=-1), lloyd_params=dict(n_samples=512),
        sliced_params=dict(), mma_params=dict(batch_size=-1),
        mhe_params=dict(batch_size=-1), verbose=verbose
    )

    tammes_d16_n288 = (288, 16, 75.5224878)
    initial_angles_d16_n288, mmd_angles_d16_n288, \
    sliced_angles_d16_n288, lloyd_angles_d16_n288, \
    mma_angles_d16_n288, mhe_angles_d16_n288, koleo_angles_d16_n288 = _run(tammes_config=tammes_d16_n288, epochs=epochs, lrs=lrs,
        mmd_params=dict(gamma=25, batch_size=-1), lloyd_params=dict(n_samples=512),
        sliced_params=dict(), mma_params=dict(batch_size=-1),
        mhe_params=dict(batch_size=-1), verbose=verbose
    )
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('legend', fontsize=14)

    #fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21,5), sharex=True)

    optimals = [tammes_n13[2], tammes_n14[2], tammes_n24[2],tammes_n128[2], tammes_d16_n288[2]]
    n_points = [13, 14, 24, 128, 288]
    dimensions = [3, 3, 3, 3, 16]
    #optimals = [tammes_n24[2]]
    #n_points = [24]

    configurations_14 = {
        "rand": initial_angles_n14,
        "mma": mma_angles_n14,
        "mhe": mhe_angles_n14,
        "koleo": koleo_angles_14,
        "mmd": mmd_angles_n14,
        "lloyd": lloyd_angles_n14,
        "sliced": sliced_angles_n14,

    }
    configurations_13 = {
        "rand": initial_angles_n13,
        "mma": mma_angles_n13,
        "mhe": mhe_angles_n13,
        "koleo": koleo_angles_n13,
        "mmd": mmd_angles_n13,
        "lloyd": lloyd_angles_n13,
        "sliced": sliced_angles_n13,
    }
    configurations_24 = {
        "rand": initial_angles_n24,
        "mma": mma_angles_n24,
        "mhe": mhe_angles_n24,
        "koleo": koleo_angles_n24,
        "mmd": mmd_angles_n24,
        "lloyd": lloyd_angles_n24,
        "sliced": sliced_angles_n24,

    }
    configurations_128 = {
        "rand": initial_angles_n128,
        "mma": mma_angles_n128,
        "mhe": mhe_angles_n128,
        "koleo": koleo_angles_n128,
        "mmd": mmd_angles_n128,
        "lloyd": lloyd_angles_n128,
        "sliced": sliced_angles_n128,
    }

    configurations_16_288 = {
        "rand": initial_angles_d16_n288,
        "mma": mma_angles_d16_n288,
        "mhe": mhe_angles_d16_n288,
        "koleo": koleo_angles_d16_n288,
        "mmd": mmd_angles_d16_n288,
        "lloyd": lloyd_angles_d16_n288,
        "sliced": sliced_angles_d16_n288,

    }

    configurations = [configurations_13, configurations_14, configurations_24,configurations_128, configurations_16_288]
    out_files = ["tammes_n13_boxplot_new.pdf", "tammes_n14_boxplot_new.pdf", "tammes_n24_boxplot_new.pdf","tammes_n128_boxplot_new.pdf","tammes_d16_n288_boxplot_new.pdf"]
    #configurations = [configurations_24]
    for j,c in enumerate(configurations):
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(14, 5), sharex=True)
        axes.boxplot(c.values(), showfliers=False,
                 medianprops=dict(color=colors.CSS4_COLORS["maroon"]))
        i = 0
        clrs = [colors.CSS4_COLORS["forestgreen"],
                colors.CSS4_COLORS["gold"],
                colors.CSS4_COLORS["royalblue"],
                colors.CSS4_COLORS["palevioletred"]]
        markers = ["o", "s", "s", "s", "D", "D", "D"]
        szs = [72, 100, 100, 100, 72, 72, 72]
        for cfg_name, vals in c.items():
            scatter_x = [i + 1 + random.uniform(-0.2, 0.2) for _ in range(len(vals))]
            axes.scatter(scatter_x, vals, s=szs[i],alpha=0.5, c=clrs[0], marker=markers[i])
            i+=1


        axes.set_xticks([y + 1 for y in range(7)],
                      labels=["None (Rand. Init.)","MMA", "MHE", "KoLeo", 'MMD','Lloyd','Sliced'])

        axes.plot([0,1,2,3,4,5,6,7,8],[optimals[j]]*9,
                  label="Optimal min. angle for d={}, N={}".format(dimensions[j],n_points[j]),
                  color='black', linestyle="--")
        axes.grid()

        axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                    ncol=1, fancybox=True, shadow=True)
        #axes.set_title(n_points[j])
        axes.set_xlabel("Optimization method")
        axes.set_ylabel("Min. angle")
        axes.set_ylim(0, 90)
        if out_file is not None:
            plt.savefig(out_files[j], bbox_inches='tight', transparent=True)

        plt.show()

    #plot
    # add_dispersed_points_to_ax(
    #     initial_angles=initial_angles_n13, mmd_angles=mmd_angles_n13,
    #     lloyd_angles=lloyd_angles_n13, sliced_angles=sliced_angles_n13,
    #     tammes_config=tammes_n13, ax=axes[0]
    # )
    # add_dispersed_points_to_ax(
    #     initial_angles=initial_angles_n14, mmd_angles=mmd_angles_n14,
    #     lloyd_angles=lloyd_angles_n14, sliced_angles=sliced_angles_n14,
    #     tammes_config=tammes_n14, ax=axes)
    # add_dispersed_points_to_ax(
    #     initial_angles=initial_angles_n24, mmd_angles=mmd_angles_n24,
    #     lloyd_angles=lloyd_angles_n24, sliced_angles=sliced_angles_n24,
    #     tammes_config=tammes_n24, ax=axes[2]
    # )




if __name__=="__main__":
    main(out_file="tammes_24_boxplt_eucl.pdf")
    #main(out_file="tammes_131424_boxplt.pdf")
