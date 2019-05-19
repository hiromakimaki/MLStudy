import numpy as np
from scipy import stats

np.random.seed(0)


def f_test(xs, ys, alpha):
    """
    Summary:
        Execute two-sample two-sided chisquare-test.
    Return value:
        0: The null hypothesis is NOT rejected.
        1: Rejected.
    """
    nx = len(xs)
    ny = len(ys)
    f = np.var(xs, ddof=1) / np.var(ys, ddof=1)
    lower, upper = stats.f.ppf(q=[alpha/2, 1-alpha/2], dfn=nx-1, dfd=ny-1)
    if f < lower or upper < f:
        return 1
    return 0


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def sample_size_1(self):
        return 50

    @property
    def sample_size_2(self):
        return 60

    def sampling(self, sample_size_1, sample_size_2):
        pass

class CaseSameVariance(TestCase):

    def sampling(self, sample_size_1, sample_size_2):
        return np.random.normal(0, 3, sample_size_1), np.random.normal(1, 3, sample_size_2)


class CaseDiffVariance(TestCase):

    def sampling(self, sample_size_1, sample_size_2):
        return np.random.normal(0, 2, sample_size_1), np.random.normal(1, 3, sample_size_2)


def simulation(test_case, n_iter=1000):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs, ys = test_case.sampling(test_case.sample_size_1, test_case.sample_size_2)
        rejected[i] = f_test(xs, ys, test_case.alpha)
    return rejected


def main():
    rejected = simulation(CaseSameVariance())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(CaseDiffVariance())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()