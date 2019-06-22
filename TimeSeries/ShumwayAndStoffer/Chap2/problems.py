import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

np.random.seed(0)


def prob_2_3():
    # (a)
    n_series = 4
    n = 100
    delta = 0.01
    sigma = 1
    ts = np.arange(n) + 1
    for i in range(n_series):
        xs = delta * ts + np.random.normal(0, sigma, size=n).cumsum() # random walk with drift
        (params, residuals, rank, s) = np.linalg.lstsq(ts.reshape((-1, 1)), xs, rcond=None)
        plt.plot(ts, xs, label='{}-th data'.format(i+1))
        plt.plot(ts, ts * params[0], label='{}-th fitted line'.format(i+1))
    plt.legend(loc='upper left')
    plt.show()


def main(args):
    func_name = 'prob_2_{}'.format(args.prob_no)
    globals()[func_name]()
    print('Fin.')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--no', dest='prob_no', type=int, default=1)
    return parser.parse_args()


if __name__=='__main__':
    main(parse_args())