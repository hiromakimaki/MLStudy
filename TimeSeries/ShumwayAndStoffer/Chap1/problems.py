import numpy as np
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


def main(args):
    if args.prob_no == 1:
        prob_1_1()
    elif args.prob_no == 2:
        prob_1_2()
    print('Fin.')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--no', dest='prob_no', type=int, default=1)
    return parser.parse_args()


if __name__=='__main__':
    main(parse_args())