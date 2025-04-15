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
    AxisAlignedBatchSphereDispersion
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
                # X = F.normalize(X, dim=-1)

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
    'lloyd': LloydSphereDispersion,
    'koleo': KoLeoDispersion,
    'mma': MMADispersion,
    'sliced': SlicedSphereDispersion,
    'sliced_axis': AxisAlignedBatchSphereDispersion
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
        # 'manif': 'exact',
        'manif': 'euclidean',
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
    ]

    for delta in deltas:
        for seed in (42,):   # 52, 62, 72, 82):
            config = base_config | delta | {'seed': seed}
            results = bench(**config, return_angles=True)
            with open('results_tammes_eucl.json', 'a') as f:
                line = json.dumps({'config': config, 'results': results})
                print(line, file=f)


def main():

    base_config = {
        'n': 20000,
        'd': 64,
        'init': 'ps100',
        # 'init': 'uniform',
        'opt': 'adam',
        'lr': 0.001,
        'manif': 'exact',
        'n_iter': 5000,
        'batch_size': None,
    }

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
                'n_samples': 13
            }
        },
    ]

    # pw complexity is n x n  x d
    # lloyd complexity is n x n_spl x d
    # sliced complexity is n x n_spl x d

    # say we pick n_spl = 10 for lloyd and sliced.
    # bsz should be about sqrt(n * 10)
    # i will take bsz=512 and n_spl = 13.

    for delta in deltas:
        for seed in (42, 52, 62):
            config = base_config | delta | {'seed': seed}
            results = bench(**config)
            with open('results.json', 'a') as f:
                line = json.dumps({'config': config, 'results': results})
                print(line, file=f)


if __name__ == '__main__':
    tammes()
    # main()
