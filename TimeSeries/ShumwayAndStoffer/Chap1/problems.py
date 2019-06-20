import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

np.random.seed(0)


def prob_1_1():
    pass


def prob_1_2():
    n = 200
    # model (a)
    signals = np.zeros(n)
    latter_t = np.arange(100, n)
    signals[latter_t] = 10 * np.exp(- (latter_t - 100)/20) * np.cos(2 * np.pi * latter_t / 4)
    noises = np.random.normal(0, 1, n)
    xs = signals + noises
    plt.plot(xs, label='model (a)')
    # model (b)
    signals = np.zeros(n)
    latter_t = np.arange(100, n)
    signals[latter_t] = 10 * np.exp(- (latter_t - 100)/200) * np.cos(2 * np.pi * latter_t / 4)
    noises = np.random.normal(0, 1, n)
    xs = signals + noises
    plt.plot(xs, label='model (b)')
    # Plot
    plt.legend(loc='upper left')
    plt.show()


def prob_1_3():
    def model_a(size):
        xs = np.zeros(size)
        for t in range(2, size):
            xs[t] = - 0.9 * xs[t-2] + np.random.normal(0, 1)
        return xs

    def model_b(size):
        return np.cos(2 * np.pi * np.arange(size) / 4)

    def model_c(size):
        return np.cos(2 * np.pi * np.arange(size) / 4) + np.random.normal(0, 1, size)

    n = 100
    window = 4
    for model in [model_a, model_b, model_c]:
        xs = model(n)
        vs = pd.Series(xs).rolling(window=window).mean().values
        '''
        # The following codes can be used for calculating moving averages:
        vs = np.zeros(n)
        for t in range(window-1, n):
            vs[t] = np.mean(xs[(t+1-window):(t+1)])
        '''
        plt.plot(np.arange(n), xs, label='x')
        plt.plot(np.arange(window-1, n), vs[(window-1):], linestyle='--', label='moving average')
        plt.title('Observation for {}'.format(model.__name__))
        plt.legend(loc='upper left')
        plt.show()


def prob_1_4():
    print('No code exists to be implemented.')


def prob_1_6():
    print('No code exists to be implemented.')


def main(args):
    func_name = 'prob_1_{}'.format(args.prob_no)
    globals()[func_name]()
    print('Fin.')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--no', dest='prob_no', type=int, default=1)
    return parser.parse_args()


if __name__=='__main__':
    main(parse_args())