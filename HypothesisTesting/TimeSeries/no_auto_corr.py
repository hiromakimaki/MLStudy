import numpy as np
from scipy import stats

np.random.seed(0)


def ljung_box_test(xs, m, alpha):
    n = len(xs)
    assert n > m
    ac = auto_corr(xs, m)
    q = n * (n + 2) * np.sum(ac / (n - np.arange(1, m+1)))
    upper = stats.chi2.ppf(q=1-alpha, df=m)
    if upper < q:
        return 1
    return 0


def auto_corr(xs, m):
    n = len(xs)
    assert n > m
    mu = np.mean(xs)
    auto_corrs = np.zeros(m+1)
    for lag in range(m+1):
        ys = xs[:n-lag]
        zs = xs[lag:]
        auto_corrs[lag] = np.mean((ys - mu) * (zs  - mu))
    auto_corrs = auto_corrs[1:] / auto_corrs[0]
    return auto_corrs


def simulation(n_iter = 500):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = np.random.normal(0, 5, 30)
        rejected[i] = ljung_box_test(xs, 8, 0.05)
    return rejected


def main():
    rejected = simulation()
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()