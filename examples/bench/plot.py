import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import json


EUCL = r' ($d_\mathbb{R}$)'
GEOD = r' ($d_\mathbb{S}$)'


def cfg_to_legend(cfg):
    if cfg['reg'] == 'mmd':
        return f"{cfg['reg']}-{cfg['args']['kernel']}-{cfg['args']['distance']}"
    else:
        return f"{cfg['reg']}-{cfg['batch_size']}"

def cfg_to_marker(cfg):
    if cfg['reg'] == 'koleo':
        return 'o'
    if cfg['reg'] == 'lloyd':
        return 's'
    if cfg['reg'] == 'sliced_axis':
        return 'x'
    return None


def get_all_runs(datas, cond, key):

    rows = []
    for data in datas:
        cfg, res = data['config'], data['results']
        if cond(cfg):
            rows.append(res[key])

    rows = np.array(rows)
    mean = rows.mean(axis=0)
    nruns = rows.shape[0]
    se = sem(rows, axis=0)
    return mean, se, nruns


def timing():
    filters = [
    {
        'name': 'MMD (Laplace, geodesic)',
        'args': {'color': 'C0', 'ls': ':', 'marker': 'v', 'markevery': 500},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'geodesic')
    },
    {
        'name': 'MMD (Laplace, euclidean)',
        'args': {'color': 'C0', 'marker': 'v', 'markevery': 500},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'MMD (gaussian, euclidean)',
        'args': {'color': 'C1', 'lw': 2},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'gaussian' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'MMD (Riesz, geodesic)',
        'args': {'color': 'C2', 'ls': ":"},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'geodesic')
    },
    {
        'name': 'MMD (Riesz, euclidean)',
        'args': {'color': 'C2'},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'KoLeo (euclidean)',
        'args': {'color': 'C3', 'marker': 'o', 'markevery': 500},
        'cond': lambda cfg: cfg['reg'] == 'koleo'
    },
    {
        'name': 'MMA (geodesic)',
        'args': {'color': 'C4', 'ls': '--'}, #, 'marker': 's', 'markevery': 120},
        'cond': lambda cfg: cfg['reg'] == 'mma'
    },
    {
        'name': 'Lloyd',
        'args': {'color': 'C5', 'lw': 3},
        'cond': lambda cfg: (cfg['reg'] == 'lloyd' and
                             cfg['batch_size'] == 512)
    },
    {
        'name': 'Sliced',
        'args': {'color': 'C6', 'marker': 'x', 'markevery': 500},
        'cond': lambda cfg: (cfg['reg'] == 'sliced_axis' and
                             cfg['batch_size'] == None)
    },
    # {
        # 'name': 'sliced-batched',
        # 'args': {'color': 'C7', 'marker': 'x', 'markevery': 100},
        # 'cond': lambda cfg: (cfg['reg'] == 'sliced_axis' and
                             # cfg['batch_size'] != None)
    # }
    ]


    fn = f'results_unif.json'
    with open(fn) as f:
        datas = [json.loads(line) for line in f]
    fn = f'results_ps100.json'
    with open(fn) as f:
        datas += [json.loads(line) for line in f]

    for filt in filters:
        lab = filt['name']
        mean, se, nruns = get_all_runs(datas, filt['cond'], key='time')
        print(f"{lab} & ${5000/mean[-1]:.2f} \\pm {se[-1]:.2f}$ \\\\")



def make_figure():

    filters = [
    {
        'name': 'MHE Laplace' + GEOD,
        'args': {'color': 'C0', 'ls': ':', 'marker': 'v', 'markevery': 500},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'geodesic')
    },
    {
        'name': 'MHE Laplace' + EUCL,
        'args': {'color': 'C0', 'marker': 'v', 'markevery': 500},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'MHE RBF' + EUCL,
        'args': {'color': 'C1', 'lw': 2},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'gaussian' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'MHE Riesz' + GEOD,
        'args': {'color': 'C2', 'ls': ":"},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'geodesic')
    },
    {
        'name': 'MHE Riesz' + EUCL,
        'args': {'color': 'C2'},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'KoLeo' + EUCL,
        'args': {'color': 'C3', 'marker': 'o', 'markevery': 500},
        'cond': lambda cfg: cfg['reg'] == 'koleo'
    },
    {
        'name': 'MM' + GEOD,
        'args': {'color': 'C4', 'ls': '--'}, #, 'marker': 's', 'markevery': 120},
        'cond': lambda cfg: cfg['reg'] == 'mma'
    },
    {
        'name': 'Lloyd',
        'args': {'color': 'C5', 'lw': 3},
        'cond': lambda cfg: (cfg['reg'] == 'lloyd' and
                             cfg['batch_size'] == 512)
    },
    {
        'name': 'Sliced',
        'args': {'color': 'C6', 'marker': 'x', 'markevery': 500},
        'cond': lambda cfg: (cfg['reg'] == 'sliced_axis' and
                             cfg['batch_size'] == None)
    },
    # {
        # 'name': 'sliced-batched',
        # 'args': {'color': 'C7', 'marker': 'x', 'markevery': 100},
        # 'cond': lambda cfg: (cfg['reg'] == 'sliced_axis' and
                             # cfg['batch_size'] != None)
    # }
    ]


    for init in ('ps100', 'unif'):
        for key in ('minds', 'cvars'):

            plt.figure(figsize=(5,4), constrained_layout=True)
            # plt.figure(figsize=(6,5), constrained_layout=True)
            fn = f'results_{init}.json'
            with open(fn) as f:
                datas = [json.loads(line) for line in f]

            for filt in filters:
                lab = filt['name']
                mean, se, nruns = get_all_runs(datas, filt['cond'], key=key)
                print(lab, mean[-5:], se[-5:], nruns)
                ix = np.arange(mean.shape[0])
                plt.plot(ix, mean, label=lab, **filt['args'])
                plt.fill_between(ix, mean-se, mean+se,
                                 color=filt['args']['color'], alpha=.1)


            if init == 'ps100':
                if key == 'cvars':
                    plt.ylabel("Spherical variance")
                elif key == 'minds':
                    plt.ylabel("Minimum geodesic distance")

            if key == 'cvars':
                if init == 'ps100':
                    plt.title(r"Clumped (PS $\kappa=100$) init")
                    plt.legend(frameon=False)
                else:
                    plt.title(r"Uniform init")

            if key == 'minds':
                plt.xlabel('Number of gradient steps')

            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            imgfn = f"toy-{init}-{key}.pdf"
            plt.savefig(imgfn)


if __name__ == '__main__':
    # timing()
    make_figure()

