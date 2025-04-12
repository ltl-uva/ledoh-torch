from time import perf_counter
import tqdm
import json

import torch
import torch.nn.functional as F

from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianSGD, RiemannianAdam
from geoopt.manifolds import SphereExact, Sphere

import matplotlib.pyplot as plt

from ledoh_torch import minimum_acos_distance, circular_variance
from ledoh_torch.pairwise_dispersion import KoLeoDispersion
from ledoh_torch import (
    KernelSphereDispersion,
    LloydSphereDispersion,
    SlicedSphereDispersion)

def _bench_one(X_init, func, make_opt, manifold, n_iter):
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    opt = make_opt(X)

    losses = []
    cvars = []
    minds = []
    time = [0.0]

    t0 = perf_counter()
    for it in tqdm.trange(n_iter):

        # log everything before update

        # update
        tic = perf_counter()
        opt.zero_grad()
        loss = func(X)

        losses.append(loss.item())
        cvars.append(circular_variance(X.detach()).item())
        minds.append(minimum_acos_distance(X.detach(), X.detach()).item())

        loss.backward()
        opt.step()
        toc = perf_counter()
        time.append(time[-1] + toc - tic)

    losses.append(func(X).item())
    cvars.append(circular_variance(X.detach()).item())
    minds.append(minimum_acos_distance(X.detach(), X.detach()).item())

    return dict(losses=losses, cvars=cvars, minds=minds, time=time)


def _get_optimizer(optimizer: str, lr: float):
    def make_opt_sgd(X):
        return RiemannianSGD([X], lr=lr)

    def make_opt_adam(X):
        return RiemannianAdam([X], lr=lr)

    if optimizer == "adam":
        return make_opt_adam
    elif optimizer == "sgd":
        return make_opt_sgd

    raise ValueError("optimizer unknown")


REGS = {
    'mmd': KernelSphereDispersion,
    'sliced': SlicedSphereDispersion,
    'lloyd': LloydSphereDispersion,
    'koleo': KoLeoDispersion,
}

INITS = {
    'uniform': lambda n, d: F.normalize(torch.randn(n, d), dim=-1)
}

MANIFS = {
    'exact': SphereExact,
}


def bench(n, d, init, reg, args, opt, lr, manif, n_iter, seed):
    torch.manual_seed(seed)
    X_init = INITS[init](n, d)
    func = REGS[reg](**args)
    manifold = MANIFS[manif]()
    make_opt = _get_optimizer(opt, lr)
    return _bench_one(X_init, func, make_opt, manifold, n_iter)


def main():

    base_config = {
        'n': 20000,
        'd': 64,
        'init': 'uniform',
        'opt': 'adam',
        'lr': 0.001,
        'manif': 'exact',
        'n_iter': 1000,
        'seed': 42,
    }

    deltas = [
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'gaussian',
                'distance': 'euclidean',
                'kernel_args': {'gamma': 1.0},
                'batch_size': 1581,
            }
        },
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'laplace',
                'distance': 'geodesic',
                'kernel_args': {'gamma': 1.0},
                'batch_size': 1581,
            }
        },
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'laplace',
                'distance': 'euclidean',
                'kernel_args': {'gamma': 1.0},
                'batch_size': 1581,
            }
        },
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'riesz',
                'distance': 'euclidean',
                'kernel_args': {'s': 1.0},
                'batch_size': 1581,
            }
        },
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'riesz',
                'distance': 'geodesic',
                'kernel_args': {'s': 1.0},
                'batch_size': 1581,
            }
        },
    ]

    for delta in deltas:

        config = base_config | delta
        results = bench(**config)
        with open('results.json', 'a') as f:
            line = json.dumps({'config': config, 'results': results})
            print(line, file=f)


if __name__ == '__main__':
    main()

