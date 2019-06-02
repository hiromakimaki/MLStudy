from scipy import stats
import numpy as np

np.random.seed(0)


def sample_data():
    return np.array([
        [9, 13, 7], # Group 1
        [22, 17, 19], # Group 2
        [5, 10, 6], # Group 3
        [16, 11, 12] # Group 4
    ])


def friedman_test(alpha, x):
    n_group, n_pair = x.shape
    rank_sum = stats.mstats.rankdata(x, axis=0).sum(axis=1)
    t = 12 * (rank_sum**2).sum() / (n_pair * n_group * (n_group + 1)) - 3 * n_pair * (n_group + 1)
    upper = stats.chi2.ppf(q=1-alpha, df=n_group-1)
    if upper < t:
        return 1
    return 0


def main():
    # Expected: rejected
    assert 1 == friedman_test(0.05, sample_data())
    print('OK')


if __name__=='__main__':
    main()