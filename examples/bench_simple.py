import math
from time import perf_counter
from functools import partial

import torch
import torch.nn.functional as F

from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianSGD, RiemannianAdam
from geoopt.manifolds import SphereExact, Sphere

import matplotlib.pyplot as plt

from ledoh_torch import minimum_acos_distance, circular_variance
from ledoh_torch import (
    KernelSphereDispersion,
    LloydSphereDispersion,
    SlicedSphereDispersion,
    AxisAlignedSlicedSphereDispersion)


def _bench_one(X_init, func, make_opt, manifold, n_iter=3000):
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    opt = make_opt(X)

    losses = []
    cvars = []
    minds = []
    time = [0.0]

    t0 = perf_counter()
    for it in range(n_iter):

        # log everything before update
        # losses.append(loss.item())
        cvars.append(circular_variance(X.detach()).item())
        minds.append(minimum_acos_distance(X.detach()).item())

        # update
        tic = perf_counter()
        opt.zero_grad()
        loss = func(X)
        loss.backward()
        opt.step()
        toc = perf_counter()
        time.append(time[-1] + toc - tic)

    cvars.append(circular_variance(X.detach()).item())
    minds.append(minimum_acos_distance(X.detach()).item())

    return dict(losses=losses, cvars=cvars, minds=minds, time=time)


def bench(n, d):
    torch.manual_seed(42)
    X_init = F.normalize(10 + torch.randn(n, d), dim=-1)

    def make_opt(X):
        return RiemannianAdam([X], lr=0.05)
        # return RiemannianSGD([X], lr=0.1)

    mmd = partial(KernelSphereDispersion.forward, gamma=.1, batch_size=-1)
    mmdmb = partial(KernelSphereDispersion.forward, gamma=.1,
                    batch_size=math.ceil(math.sqrt(n)))
    lloyd = partial(LloydSphereDispersion.forward,
                    n_samples=math.ceil(math.log10(n)))
    sliced = partial(SlicedSphereDispersion.forward, p=None, q=None)
    axsliced = partial(AxisAlignedSlicedSphereDispersion.forward, i=None, j=None)

    res = dict()

    if n <= 100:
        res["mmd-full"] = _bench_one(X_init=X_init,
                                func=mmd,
                                make_opt=make_opt,
                                manifold=Sphere())

    res["mmd-mb"] = _bench_one(X_init=X_init,
                               func=mmdmb,
                               make_opt=make_opt,
                               manifold=Sphere())

    res["lloyd"] = _bench_one(X_init=X_init,
                              func=lloyd,
                              make_opt=make_opt,
                              manifold=Sphere())

    res["sliced"] = _bench_one(X_init=X_init,
                               func=sliced,
                               make_opt=make_opt,
                               manifold=Sphere())
    res["sliced-ax"] = _bench_one(X_init=X_init,
                                  func=axsliced,
                                  make_opt=make_opt,
                                  manifold=Sphere())

    return res


def main():

    d = 16

    pltargs = {
        'mmd-full': {'color': 'C0'},
        'mmd-mb': {'color': 'C0', 'ls': ":"},
        'lloyd': {'color': 'C2'},
        'sliced': {'color': 'C3'},
        'sliced-ax': {'color': 'C3', 'ls': ":"},
    }


    fig, axes = plt.subplots(nrows=4, ncols=3)
    for i, n in enumerate((100, 1000, 2000)):

        res = bench(n, d)

        for method, scores in res.items():
            axes[0, i].plot(scores["cvars"], label=method, **pltargs[method])
            axes[1, i].plot(scores["minds"], label=method, **pltargs[method])
            axes[2, i].plot(scores["time"], scores["cvars"], label=method,
                            **pltargs[method])
            axes[3, i].plot(scores["time"], scores["minds"], label=method,
                            **pltargs[method])

        axes[0, i].set_title(f"{n=}")
        axes[0, i].set_xlabel("iter")
        axes[1, i].set_xlabel("iter")
        axes[2, i].set_xlabel("time")
        axes[3, i].set_xlabel("time")

    axes[0, 0].legend()

    axes[0, 0].set_ylabel("cvar")
    axes[1, 0].set_ylabel("min d")
    axes[2, 0].set_ylabel("cvar")
    axes[3, 0].set_ylabel("min d")

    plt.show()


if __name__ == '__main__':
    main()

