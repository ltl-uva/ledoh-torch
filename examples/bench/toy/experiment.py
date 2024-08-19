import json
from typing import Callable, Dict, List
from time import perf_counter
from functools import partial
import argparse

from tqdm import tqdm

import torch
import torch.nn.functional as F

import numpy as np

from power_spherical.distributions import PowerSpherical

from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianSGD, RiemannianAdam
from geoopt.manifolds import SphereExact

from ledoh_torch import (
    KernelSphereDispersion,
    KernelSphereSemibatchDispersion,
    LloydSphereDispersion,
    AxisAlignedBatchSphereDispersion,
    SlicedSphereDispersion,
    MMADispersion,
    minimum_acos_distance as minimum_acos_distance,
    circular_variance
)

import pandas as pd

from bench_utils import ExperimentConfig


def _get_optimizer(optimizer: str, lr: float) -> Callable:
    def make_opt_sgd(X):
        return RiemannianSGD([X], lr=lr)

    def make_opt_adam(X):
        return RiemannianAdam([X], lr=lr)

    if optimizer == "adam":
        return make_opt_adam
    elif optimizer == "sgd":
        return make_opt_sgd

    raise ValueError("optimizer unknown")


def _get_init_embeddings(n: int, d: int, init: dict, device):
    X_init = None

    if init["_name"] == "gaussian":
        offset_init = 10
        torch.manual_seed(42)
        X_init = F.normalize(offset_init + torch.randn(n, d), dim=-1).to(device)

    # elif "vmf" == init[:3]:
    #     mu = np.zeros(d)
    #     mu[0] = 1
    #     kappa = int(init[3:])
    #     X_init = F.normalize(torch.from_numpy(vonmises_fisher(mu, kappa).rvs(n, random_state=42)), dim=-1).to(device).to(torch.float32)
    #     return X_init
    elif init["_name"]=="powerspherical_constant":
        kappa = init["kappa"]
        ps_dist = PowerSpherical(
            F.normalize(torch.full((n, d), d ** -0.5, dtype=torch.float32,device=device), dim=-1),
            scale=torch.tensor(kappa, dtype=torch.float32, device=device).repeat(n))
        X_init = ps_dist.rsample()

    elif init["_name"]=="powerspherical_decay":
        init_kappa_val = init["kappa"]
        stop_decay = 1000
        kappa = torch.linspace(init_kappa_val, stop_decay, n, dtype=torch.float32, device=device)
        ps_dist = PowerSpherical(
            F.normalize(torch.full((n, d), d ** -0.5, dtype=torch.float32,device=device), dim=-1),
            scale=kappa)
        X_init = ps_dist.rsample()

    X_init_ = X_init.detach().clone()
    init_min_dist = minimum_acos_distance(X_init_, X_init_)
    circular_variance_init = circular_variance(X_init_)
    print(f"init min dist {init_min_dist}, init cvar {circular_variance_init}")
    return X_init


def _bench_one(
        X_init: torch.Tensor, func: Callable,
        make_opt: Callable, manifold: SphereExact,
        n_iter: int,
        device: torch.cuda.device
    ) -> Dict[str, List[float]]:
    X = ManifoldParameter(X_init.clone(), manifold=manifold)
    opt = make_opt(X)
    X = X.to(device)

    losses = []
    cvars = []
    minds = []
    time = [0.0]

    t0 = perf_counter()

    for it in tqdm(range(n_iter)):
        tic = perf_counter()
        opt.zero_grad()
        loss = func(X)
        loss.backward()
        opt.step()
        toc = perf_counter()
        time.append(time[-1] + toc - tic)

        losses.append(loss.detach().item())
        cvars.append(circular_variance(X.detach()).item())
        minds.append(minimum_acos_distance(X.detach(),X.detach()).item())

        print(f"iter {it} loss {losses[-1]} cvar {cvars[-1]} mind {minds[-1]} time {time[-1]}")

    return X.detach(), dict(losses=losses, cvars=cvars, minds=minds, time=time[1:])


def main(config: ExperimentConfig):
    # i didnt add the mps device btw
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #project_name = config.project_name
    #logger = WandbLogger(config)


    for lr, n, d, n_iter in config.get_hyper_params():
        # create embeddings
        init_method = config.init_embeddings
        X_init = _get_init_embeddings(n=n, d=d, init=init_method, device=device)
        print(n,d, X_init.shape)

        for model_name, params, optim_type in config.get_model():
            # logger.start_run(
            #     project_name, model_name, init_method, (lr, n, d, n_iter), params, optim_type
            # )
            # prepare model
            if model_name == "mmd":
                loss_fn = KernelSphereDispersion(**params)
            elif model_name == "sliced":
                loss_fn = SlicedSphereDispersion()
            elif model_name == "sliced-ax":
                loss_fn = AxisAlignedBatchSphereDispersion(**params)
            elif model_name == "lloyd":
                loss_fn = LloydSphereDispersion(**params)
            elif model_name == "mma":
                loss_fn = MMADispersion(**params)
            elif model_name == "mmd-semi":
                loss_fn = KernelSphereSemibatchDispersion(**params)
            else:
                raise Exception("Incorrect model specified in configuration")

            make_opt = _get_optimizer(optim_type, lr)
            manifold = SphereExact()

            embeddings, results = _bench_one(X_init=X_init,
                              func=loss_fn,
                              make_opt=make_opt,
                              manifold=manifold,
                            n_iter=n_iter, device=device)


            #logger.log(embeddings, results, finish=True)
            pd.DataFrame(results).to_csv(f"{model_name}_{init_method['_name']}_{lr}_{n}_{d}_{n_iter}_{optim_type}_{json.dumps(params)}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="path to configuration file",
        type=str
    )
    args = parser.parse_args()
    config = ExperimentConfig(args.config)

    main(config)

