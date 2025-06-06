import os
from time import perf_counter
import tqdm
import json

import torch
import torch.nn.functional as F

from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianSGD, RiemannianAdam
from geoopt.manifolds import SphereExact, Euclidean
from power_spherical.distributions import PowerSpherical

import matplotlib.pyplot as plt

from ledoh_torch import (
    minimum_acos_distance,
    circular_variance,
    get_acos_distance_matrix
)
from ledoh_torch import (
    KernelSphereDispersion,
    LloydSphereDispersion,
    KoLeoDispersion,
    MMADispersion,
    SlicedSphereDispersion,
    AxisAlignedBatchSphereDispersion,
    SphericalSlicedWassersteinDispersion
)

def _bench_one(X_init, func, make_opt, manifold, batch_size, n_iter, seed,
               return_angles=False):
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    opt = make_opt(X)

    losses = []
    cvars = []
    minds = []
    time = [0.0]

    torch.manual_seed(seed)

    t0 = perf_counter()
    for it in tqdm.trange(n_iter):

        # log everything before update

        # update
        tic = perf_counter()
        opt.zero_grad()

        X_mb = X
        if batch_size is not None:
            ix = torch.randperm(X.shape[0], device=X.device)[:batch_size]
            X_mb = X_mb[ix]
        loss = func(X_mb)

        losses.append(loss.item())
        cvars.append(circular_variance(X.detach()).item())
        minds.append(minimum_acos_distance(X.detach(), X.detach()).item())

        loss.backward()
        opt.step()
        toc = perf_counter()

        if isinstance(manifold, Euclidean):
            with torch.no_grad():
                X.data = F.normalize(X.data, dim=-1)

        time.append(time[-1] + toc - tic)

    losses.append(func(X).item())
    cvars.append(circular_variance(X.detach()).item())
    minds.append(minimum_acos_distance(X.detach(), X.detach()).item())
    ret = dict(losses=losses, cvars=cvars, minds=minds, time=time)

    if return_angles:
        Xd = X.detach()
        n = Xd.shape[0]
        angles = get_acos_distance_matrix(Xd, Xd)
        angles = angles.fill_diagonal_(torch.inf)
        ret['angles'] = angles.min(dim=1)[0].tolist()

    return ret


def _get_optimizer(optimizer: str, lr: float):
    def make_opt_sgd(X):
        return RiemannianSGD([X], lr=lr, stabilize=1)

    def make_opt_adam(X):
        return RiemannianAdam([X], lr=lr, stabilize=1)

    if optimizer == "adam":
        return make_opt_adam
    elif optimizer == "sgd":
        return make_opt_sgd

    raise ValueError("optimizer unknown")


REGS = {
    'mmd': KernelSphereDispersion,
    'lloyd': LloydSphereDispersion,
    'koleo': KoLeoDispersion,
    'mma': MMADispersion,
    'sliced': SlicedSphereDispersion,
    'sliced_axis': AxisAlignedBatchSphereDispersion,
    'ssw': SphericalSlicedWassersteinDispersion
}

INITS = {
    'uniform': lambda n, d: F.normalize(torch.randn(n, d), dim=-1),
    'ps100': lambda n, d: PowerSpherical(
        loc=F.normalize(torch.randn(d), dim=-1),
        scale=torch.tensor(100.0)
    ).rsample((n,))
}

MANIFS = {
    'euclidean': Euclidean,
    'exact': SphereExact,
}


def bench(
    n,
    d,
    init,
    reg,
    args,
    opt,
    lr,
    manif,
    n_iter,
    seed,
    batch_size,
    return_angles=False
    ):

    print(reg, args)

    torch.manual_seed(seed)
    X_init = INITS[init](n, d)

    if torch.cuda.is_available():
        X_init = X_init.cuda()

    return _bench_one(
        X_init=X_init,
        func=REGS[reg](**args),
        make_opt=_get_optimizer(opt, lr),
        manifold=MANIFS[manif](),
        batch_size=batch_size,
        n_iter=n_iter,
        seed=seed+1,  # just to be sure, use deterministic but different seed
        return_angles=return_angles
    )

def tammes() -> None:

    base_config = {
        'n': 24,
        'd': 3,
        'init': 'uniform',
        'opt': 'adam',
        'lr': 0.005,
        'manif': 'exact',
        # 'manif': 'euclidean',
        'n_iter': 10000,
        'batch_size': None,
    }

    deltas = [
        {
            'reg': 'mma',
            'args': {},
            'n_iter': 0
        },
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'laplace',
                'distance': 'geodesic',
                'kernel_args': {'gamma': 1.0},
            }
        },
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'gaussian',
                'distance': 'euclidean',
                'kernel_args': {'gamma': 1.0},
            }
        },
        {
            'reg': 'mmd',
            'args': {
                'kernel': 'riesz',
                'distance': 'euclidean',
                'kernel_args': {'s': 1.0},
            }
        },
        {
            'reg': 'mma',
            'args': {}
        },
        {
            'reg': 'koleo',
            'args': {}
        },
        {
            'reg': 'lloyd',
            'args': {
                'n_samples': 300
            }
        },
        {
            'reg': 'sliced',
            'args': {
            }
        },
        {
            'reg': 'sliced_axis',
            'args': {'n_samples':24,
            }
        },
        {
            'reg': 'ssw',
            'args': {'n_projections':50},
        }
    ]

    if os.path.exists('results_tammes.json'):
        os.remove('results_tammes.json')

    for delta in deltas:
        for seed in (42,):   # 52, 62, 72, 82):
            config = base_config | delta | {'seed': seed}
            results = bench(**config, return_angles=True)
            with open('results_tammes.json', 'a') as f:
                line = json.dumps({'config': config, 'results': results})
                print(line, file=f)


def main(n,d, lr=0.001, niter=5000, sn_samples=None):

    base_config = {
        'n': n,
        'd': d,
        'init': 'ps100',
        # 'init': 'uniform',
        'opt': 'adam',
        'lr':lr,
        'manif': 'exact',
        'n_iter': niter,
        'batch_size': None,
    }
    if sn_samples is None:
        sn_samples = int(round(512**2/n))

    deltas = [
        {
            'batch_size': 512,
            'reg': 'mmd',
            'args': {
                'kernel': 'laplace',
                'distance': 'geodesic',
                'kernel_args': {'gamma': 1.0},
            }
        },
        {
            'batch_size': 512,
            'reg': 'mmd',
            'args': {
                'kernel': 'gaussian',
                'distance': 'euclidean',
                'kernel_args': {'gamma': 1.0},
            }
        },
        {
            'batch_size': 512,
            'reg': 'mmd',
            'args': {
                'kernel': 'laplace',
                'distance': 'euclidean',
                'kernel_args': {'gamma': 1.0},
            }
        },
        {
            'batch_size': 512,
            'reg': 'mmd',
            'args': {
                'kernel': 'riesz',
                'distance': 'euclidean',
                'kernel_args': {'s': 1.0},
            }
        },
        {
            'batch_size': 512,
            'reg': 'mmd',
            'args': {
                'kernel': 'riesz',
                'distance': 'geodesic',
                'kernel_args': {'s': 1.0},
            }
        },
        {
            'batch_size': 512,
            'reg': 'mma',
            'args': {}
        },
        {
            'batch_size': 512,
            'reg': 'koleo',
            'args': {}
        },
        {
            'reg': 'lloyd',
            'batch_size': 512,
            'args': {
                'n_samples': 512
            }
        },
        {
            'reg': 'lloyd',
            'args': {
                'n_samples': 13
            }
        },
        {
            'reg': 'sliced_axis',
            'batch_size': 512,
            'args': {
                'n_samples': 512
            }
        },
        {
            'reg': 'sliced_axis',
            'args': {
                'n_samples': sn_samples
            }
        },
        {
            'reg': 'ssw',
            'args': {'n_projections': sn_samples},
        }
    ]

    # pw complexity is n x n  x d
    # lloyd complexity is n x n_spl x d
    # sliced complexity is n x n_spl x d

    # say we pick n_spl = 10 for lloyd and sliced.
    # bsz should be about sqrt(n * 10)
    # i will take bsz=512 and n_spl = 13.
    if os.path.exists(f'eesults_{d}_{n}_{lr}_{sn_samples}_full.json'):
        os.remove(f'esults_{d}_{n}_{lr}_{sn_samples}_full.json')

    for delta in deltas:
        for seed in (42, 52, 62):  #, 52, 62):
            config = {**base_config, **delta, **{'seed': seed}}
            results = bench(**config)
            with open(f'results_{d}_{n}_{lr}_{sn_samples}_full.json', 'a') as f:
                line = json.dumps({'config': config, 'results': results})
                print(line, file=f)


def grid_search_ssw():
    ns = [1000]
    ds = [64,512,1024]
    lrs = [0.0001, 0.001, 0.01, 0.1]
    nsamples = [1]

    for n in ns:
        for d in ds:
            for lr in lrs:
                for n_samples in nsamples:
                    print(f'Running with n={n}, d={d}, lr={lr}, n_samples={n_samples}')
                    main(n, d, lr=lr, sn_samples=n_samples)


if __name__ == '__main__':
    #tammes()
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmarks for various regularization methods.')
    parser.add_argument('--tammes', action='store_true', help='Run benchmarks for Tammes problem.')
    parser.add_argument('--main', action='store_true', help='Run main benchmarks.')
    parser.add_argument('--n', type=int, default=20000, help='Number of points to generate.')
    parser.add_argument('--d', type=int, default=64, help='Dimensionality of the points.')
    parser.add_argument('--iters', type=int, default=5000, help='Dimensionality of the points.')
    args = parser.parse_args()

    if args.tammes:
        tammes()
    if args.main:
        main(args.n, args.d, lr=0.001, niter=args.iters)
    #grid_search_ssw()
