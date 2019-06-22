import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

np.random.seed(0)


def prob_2_3():
    def model_a(ts):
        # random walk with drift
        return 0.01 * ts + np.random.normal(0, 1, size=len(ts)).cumsum()
    def model_b(ts):
        # trend plus noise
        return 0.01 * ts + np.random.normal(0, 1, size=len(ts))
    n_series = 4
    n = 100
    ts = np.arange(n) + 1
    for model in [model_a, model_b]:
        for i in range(n_series):
            xs =  model(ts)
            (params, residuals, rank, s) = np.linalg.lstsq(ts.reshape((-1, 1)), xs, rcond=None)
            plt.plot(ts, xs, label='{}-th data'.format(i+1))
            plt.plot(ts, ts * params[0], label='{}-th fitted line'.format(i+1), linestyle='--')
        plt.legend(loc='upper left')
        plt.title('Data and fitted lines for {}'.format(model.__name__))
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