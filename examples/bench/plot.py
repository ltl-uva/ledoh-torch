import matplotlib.pyplot as plt
import json


def cfg_to_legend(cfg):
    if cfg['reg'] == 'mmd':
        return f"{cfg['reg']}-{cfg['args']['kernel']}-{cfg['args']['distance']}"
    else:
        return cfg['reg']


def main():

    plt.figure()

    with open('results.json') as f:
        for line in f:
            data = json.loads(line)
            cfg = data['config']
            res = data['results']
            lab = cfg_to_legend(cfg)
            # plt.plot(res['losses'], label=lab)
            # plt.plot(res['minds'], label=lab)
            series = res['cvars']
            # series = res['minds']
            marker = 'o' if cfg['reg'] == 'koleo' else None
            plt.plot(series, label=lab, marker=marker, markevery=100)
            print(lab)
            print(len(series))
            print(series[:10], series[-10:])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

