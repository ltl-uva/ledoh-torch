import matplotlib.pyplot as plt
import json


def main():

    plt.figure()

    with open('results.json') as f:
        for line in f:
            data = json.loads(line)
            cfg = data['config']
            res = data['results']
            lab = f"{cfg['reg']}-{cfg['args']['kernel']}-{cfg['args']['distance']}"
            # plt.plot(res['losses'], label=lab)
            # plt.plot(res['minds'], label=lab)
            # plt.plot(res['cvars'], label=lab)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

