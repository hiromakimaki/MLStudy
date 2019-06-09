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


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def max_lag(self):
        return 3

    @property
    def sample_size(self):
        return 40

    def sampling(self, sample_size):
        pass

class CaseNormalIID(TestCase):

    def sampling(self, sample_size):
        return np.random.normal(0, 5, sample_size)


class CasePeriodical(TestCase):

    def sampling(self, sample_size):
        return np.sin(np.pi / 2 * np.arange(sample_size)) + np.random.normal(0, 0.7, sample_size)


def simulation(test_case, n_iter = 500):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = test_case.sampling(test_case.sample_size)
        rejected[i] = ljung_box_test(xs, test_case.max_lag, test_case.alpha)
    print('Auto corr: {}'.format(auto_corr(xs, test_case.max_lag)))
    return rejected


def main():
    for test_case in [CaseNormalIID(), CasePeriodical()]:
        rejected = simulation(test_case)
        print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()