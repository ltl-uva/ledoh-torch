import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import json


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
    se = sem(rows, axis=0)
    return mean, se


def main():

    filters = [
    {
        'name': 'mmd-laplace-geodesic',
        'args': {'color': 'C0'},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'geodesic')
    },
    {
        'name': 'mmd-laplace-euclidean',
        'args': {'color': 'C0', 'ls': ":"},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'euclidean')
    },
    # {
        # 'name': 'mmd-gaussian-geodesic',
        # 'args': {'color': 'C1'},
        # 'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             # cfg['args']['kernel'] == 'gaussian' and
                             # cfg['args']['distance'] == 'geodesic')
    # },
    {
        'name': 'mmd-gaussian-euclidean',
        'args': {'color': 'C1', 'ls': ":"},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'gaussian' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'mmd-riesz-geodesic',
        'args': {'color': 'C2'},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'geodesic')
    },
    {
        'name': 'mmd-riesz-euclidean',
        'args': {'color': 'C2', 'ls': ':'},
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'koleo',
        'args': {'color': 'C3', 'marker': 'o', 'markevery': 100},
        'cond': lambda cfg: cfg['reg'] == 'koleo'
    },
    {
        'name': 'mma',
        'args': {'color': 'C3', 'marker': 's', 'markevery': 120},
        'cond': lambda cfg: cfg['reg'] == 'mma'
    },
    {
        'name': 'lloyd',
        'args': {'color': 'C5'},
        'cond': lambda cfg: (cfg['reg'] == 'lloyd' and
                             cfg['batch_size'] == 512)
    },
    {
        'name': 'sliced',
        'args': {'color': 'C6', 'marker': 'x', 'markevery': 100},
        'cond': lambda cfg: (cfg['reg'] == 'sliced_axis' and
                             cfg['batch_size'] == None)
    }
    ]

    plt.figure()

    # fn = 'results_unif.json'
    fn = 'results_ps100.json'
    # key = 'minds'
    key = 'cvars'
    with open(fn) as f:
        datas = [json.loads(line) for line in f]


    for filt in filters:
        lab = filt['name']
        mean, se = get_all_runs(datas, filt['cond'], key=key)
        print(lab, mean)
        ix = np.arange(mean.shape[0])
        plt.plot(ix, mean, label=lab, **filt['args'])
        plt.fill_between(ix, mean-se, mean+se,
                         color=filt['args']['color'], alpha=.1)


            # data = json.loads(line)
            # cfg = data['config']
            # if cfg['seed'] != 42:
                # continue
            # res = data['results']
            # lab = cfg_to_legend(cfg)
            # marker = cfg_to_marker(cfg)
            # # plt.plot(res['losses'], label=lab)
            # # plt.plot(res['minds'], label=lab)
            # series = res['cvars']
            # # series = res['minds']
            # plt.plot(series, label=lab, marker=marker, markevery=100)
            # print(lab)
            # print(cfg)
            # print(len(series))
            # print(series[:10], series[-10:])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

