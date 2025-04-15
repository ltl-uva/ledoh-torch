import numpy as np
import matplotlib.pyplot as plt
import json

def get_angles_all_runs(datas, cond):

    angles = []
    for data in datas:
        cfg, res = data['config'], data['results']
        if cond(cfg):
            angles.extend(res['angles'])
    angles = np.array(angles) * 180 / np.pi
    return angles


def main():
    filters = [
    {
        'name': 'none\n(unif. init.)',
        'cond': lambda cfg: cfg['n_iter'] == 0
    },
    {
        'name': 'MMD Laplace\n(geodesic)',
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'geodesic')
    },
    # {
        # 'name': 'MMD (Laplace, euclidean)',
        # 'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             # cfg['args']['kernel'] == 'laplace' and
                             # cfg['args']['distance'] == 'euclidean')
    # },
    {
        'name': 'MMD gaussian\n(euclidean)',
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'gaussian' and
                             cfg['args']['distance'] == 'euclidean')
    },
    # {
        # 'name': 'MMD (Riesz, geodesic)',
        # 'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             # cfg['args']['kernel'] == 'riesz' and
                             # cfg['args']['distance'] == 'geodesic')
    # },
    {
        'name': 'MMD Riesz\n(euclidean)',
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'KoLeo\n(euclidean)',
        'cond': lambda cfg: cfg['reg'] == 'koleo'
    },
    {
        'name': 'MMA\n(geodesic)',
        'cond': lambda cfg: (cfg['reg'] == 'mma' and cfg['n_iter'] != 0)
    },
    {
        'name': 'Lloyd',
        'cond': lambda cfg: cfg['reg'] == 'lloyd'
    },
    {
        'name': 'Sliced',
        'cond': lambda cfg: cfg['reg'] == 'sliced'
    },
    ]

    fn = f'results_tammes.json'
    with open(fn) as f:
        datas = [json.loads(line) for line in f]


    names = [filt['name'] for filt in filters]
    series = [get_angles_all_runs(datas, filt['cond']) for filt in filters]

    plt.figure(figsize=(10,3), constrained_layout=True)
    rng = np.random.default_rng(42)
    for i, angles in enumerate(series):
        jitter = rng.uniform(-.3, +.3, size=angles.shape)
        plt.scatter(jitter + i + 1, angles, marker='.', alpha=.1, color='k')

    plt.boxplot(series, sym='')
    plt.axhline(48.5352, color='k')
    plt.xticks(ticks=np.arange(1, len(names)+1), labels=names)
    plt.ylabel("minimum angle (degrees)")
    plt.savefig("toy_tammes.pdf")
    # plt.show()


def riem_vs_eucl():
    filters = [
    {
        'name': 'none\n(unif. init.)',
        'cond': lambda cfg: cfg['n_iter'] == 0
    },
    {
        'name': 'MMD Laplace\n(geodesic)',
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'laplace' and
                             cfg['args']['distance'] == 'geodesic')
    },
    {
        'name': 'MMD gaussian\n(euclidean)',
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'gaussian' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'MMD Riesz\n(euclidean)',
        'cond': lambda cfg: (cfg['reg'] == 'mmd' and
                             cfg['args']['kernel'] == 'riesz' and
                             cfg['args']['distance'] == 'euclidean')
    },
    {
        'name': 'KoLeo\n(euclidean)',
        'cond': lambda cfg: cfg['reg'] == 'koleo'
    },
    {
        'name': 'MMA\n(geodesic)',
        'cond': lambda cfg: (cfg['reg'] == 'mma' and cfg['n_iter'] != 0)
    },
    {
        'name': 'Lloyd',
        'cond': lambda cfg: cfg['reg'] == 'lloyd'
    },
    {
        'name': 'Sliced',
        'cond': lambda cfg: cfg['reg'] == 'sliced'
    },
    ]

    fn = f'results_tammes.json'
    with open(fn) as f:
        datas_riem = [json.loads(line) for line in f]
        datas_riem = [d for d in datas_riem if d['config']['seed']==42]

    fn = f'results_tammes_eucl.json'
    with open(fn) as f:
        datas_eucl = [json.loads(line) for line in f]

    names = [filt['name'] for filt in filters]
    series_riem = [get_angles_all_runs(datas_riem, filt['cond']) for filt in filters]
    series_eucl = [get_angles_all_runs(datas_eucl, filt['cond']) for filt in filters]

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), constrained_layout=True)

    rng = np.random.default_rng(42)
    for i in range(len(names)):
        angles_eucl = series_eucl[i]
        angles_riem = series_riem[i]
        print(angles_eucl)
        jitter = rng.uniform(-.3, +.3, size=angles_eucl.shape)
        ax1.scatter(jitter + i + 1, angles_eucl, marker='.', alpha=.1, color='k')
        ax2.scatter(jitter + i + 1, angles_riem, marker='.', alpha=.1, color='k')

    ax1.boxplot(series_eucl, sym='')
    ax2.boxplot(series_riem, sym='')
    ax1.axhline(48.5352, color='k')
    ax2.axhline(48.5352, color='k')
    ax1.set_ylim(0, 60)
    ax2.set_ylim(0, 60)
    plt.xticks(ticks=np.arange(1, len(names)+1), labels=names)
    plt.ylabel("minimum angle (degrees)")
    plt.savefig("toy_tammes_riem_vs_eucl.pdf")
    # plt.show()


if __name__ == '__main__':
    # riem_vs_eucl()
    main()

