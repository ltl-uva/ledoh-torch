"""
This file contains 1) a wrapper to iterate over hyperparam combinations and 2) a wandb logger
"""

from typing import Any, Dict, Tuple
from itertools import product

from ledoh_torch import minimum_acos_distance_row

import wandb
import yaml


class ExperimentConfig():
    """
    Main usage of this class is to provide iterators for looping
    through all hyperparam combinations for training and
    uploading plots to wandb
    """
    def __init__(self, path: str) -> None:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        self.project_name = config["project_name"]
        self.n_values = config["n"]
        self.lr = config["lr"]
        self.dimensions = config["d"]
        self.n_iter = config["n_iter"]
        self.models = config["models"]
        self.interval = config["interval"]
        self.init_embeddings = config["init"]

    def get_hyper_params(self):
        for n, d in product(self.n_values, self.dimensions):
            yield self.lr, n, d, self.n_iter

    def get_model(self):
        """ Creates generator to iterate through all models and their parameter confighuration """
        for model_name, properties in self.models.items():
            params = properties["params"]
            optim = properties["optim"]

            # handle edge case where there are no parameters
            if len(params) == 0:
                yield model_name, params, optim
            else:
                # create combinations of parameters
                keys = list(params.keys())
                values = [params[key] for key in keys]

                for param_values in product(*values):
                    param_combination = {
                        keys[i]: param_values[i] for i in range(len(keys))
                    }
                    yield model_name, param_combination, optim

    def get_plot_combinations(self, model_name: str) -> Tuple[int, int, Any]:
        param_names = []

        if not model_name == "hyperparams":
            params = self.models[model_name]["params"]
            param_names = [key for key, value in params.items() if len(value) > 1]
            param_names = sorted(param_names)

        # maybe not optimal to return var number of values, but it is convenient here
        return "n", "d", *param_names


class WandbLogger():
    """ Tracks training and logs data to wandb """
    def __init__(self, config: ExperimentConfig) -> None:
        self.current_run = None
        self.config = config
        wandb.login()

    def start_run(self,
                  project_name: str,
                  model_name: str,
                  init_mode: str,
                  hyperparams: Tuple[float, int, int, int],
                  params: Dict[str, Any],
                  optim_type: str
        ) -> None:
        self.model_name = model_name
        lr, n, d, n_iter = hyperparams
        run_config = {
                "regularizer": model_name,
                "learning_rate": lr,
                "epochs": n_iter,
                "d": d,
                "params": params,
                "n": n,
                "optim": optim_type,
                "init": init_mode
        }

        if "batch_size" in params:
            run_config["batch_size"] = params["batch_size"]

        run_name = f"{model_name} "
        run_name += ", ".join([f"{param}={value}" for param, value in params.items()])
        run_name += f" n={n}, d={d} optim={optim_type}"
        run_name += f" init={init_mode}"
        wandb.init(project=project_name, name=run_name, config=run_config)

        self.current_run = run_config

    def log(self, embeddings, results: Dict[str, Any], finish: bool = False) -> None:
        """ Log data to wandb """
        wandb.log(results)
        self._create_wandb_plots(results)
        self._log_min_dist(embeddings)

        if finish:
            self.finish_run()

    def finish_run(self) -> None:
        wandb.finish()
        self.current_run = None

    def _create_data_tables(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """ Create tables for spherical variance, mind dist and loss """
        def create_table(values, columns) -> wandb.Table:
            data = [[i, val] for i, val in enumerate(values) if i % interval == 0]
            return wandb.Table(data=data, columns=columns)

        interval = self.config.interval
        epoch_cvar_table = create_table(results["cvars"], columns=["epoch", "cvar"])
        epoch_dmin_table = create_table(results["minds"], columns=["epoch", "dmin"])
        epoch_loss_table = create_table(results["losses"], columns=["epoch", "loss"])
        aggregate_tables = dict(epoch=dict(cvar=epoch_cvar_table, dmin=epoch_dmin_table, loss=epoch_loss_table))
        return aggregate_tables

    def _create_plots(
            self, model_name: str, tables: Dict[str, Dict[str, Any]],
            n: int, d: int, params_values: Dict[str, Any]
        ) -> Dict[str, wandb.plot.line]:
        data = dict()

        def add_plots(section, variables, param_values, fix_all=False):
            its = len(variables) if not fix_all else 1

            for i in range(its):
                # this is part of old code
                if not fix_all:
                    fixed_vars = variables[:i] + variables[i + 1:]
                    non_fixed_var = variables[i]
                else:
                    fixed_vars = variables
                    non_fixed_var = "regularizers"

                ids = [f"{var}{param_values[var]}" for var in fixed_vars]
                fixed_ids = "_".join(fixed_vars)
                id = f"[{non_fixed_var}]" + "_".join(ids)

                # upload line plots to wandb
                for cat in ["epoch"]:
                    data[f"{section}_{fixed_ids}/{cat}_cvar_{id}"] = wandb.plot.line(
                        tables[cat]["cvar"], cat, "cvar",
                        title=f"Circular variance " + id
                    )
                    data[f"{section}_{fixed_ids}/{cat}_loss_{id}"] = wandb.plot.line(
                        tables[cat]["loss"], cat, "loss",
                        title=f"Regularizer loss " + id
                    )
                    data[f"{section}_{fixed_ids}/{cat}_dmin_{id}"] = wandb.plot.line(
                        tables[cat]["dmin"], cat, "dmin",
                        title=f"Minimum distance " + id
                    )

        # general plots
        variables = self.config.get_plot_combinations("hyperparams")
        hyperparams = dict(n=n, d=d)
        add_plots("general", variables, dict(n=n, d=d), fix_all=True)

        # model specific plots
        variables = self.config.get_plot_combinations(model_name)
        param_values_adj = params_values.copy()
        param_values_adj.update(hyperparams)
        add_plots(model_name, variables, param_values_adj)

        return data


    def _create_wandb_plots(self, results: Dict[str, Any]) -> None:
        model_name = self.current_run["regularizer"]
        n = self.current_run["n"]
        d = self.current_run["d"]
        params = self.current_run["params"]
        tables = self._create_data_tables(results)

        data = self._create_plots(
            model_name=model_name,
            tables=tables,
            n=n,
            d=d,
            params_values=params
        )

        wandb.log(data)

    def _log_min_dist(self, embeddings):
        """ Uploads table containing min dist per embedding to wandb """
        min_dist = minimum_acos_distance_row(embeddings, embeddings).unsqueeze(-1).tolist()
        table = wandb.Table(data=min_dist, columns=["dmin"])
        wandb.log({f'{self.model_name}/dmin_hist': table})
