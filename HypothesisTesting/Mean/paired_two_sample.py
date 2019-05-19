import numpy as np
from scipy import stats

np.random.seed(0)


def t_test(xs, ys, alpha):
    """
    Summary:
        Execute paired two-sample two-sided t-test.
        It is supposed that the values of (unknown) variances of the two samples are same.
    Return value:
        0: The null hypothesis is NOT rejected.
        1: Rejected.
    """
    assert len(xs) == len(ys)
    n = len(xs)
    ds = xs - ys
    t_value = np.sqrt(n) * np.mean(ds) / np.std(ds, ddof=1)
    lower, upper = stats.t.ppf(q=[alpha/2, 1-alpha/2], df=n-1)
    if t_value < lower or upper < t_value:
        return 1
    return 0


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def sample_size(self):
        return 50

    def sampling(self, sample_size):
        pass


class CaseSameMeanSameVar(TestCase):

    def sampling(self, sample_size):
        return np.random.normal(3, 1, sample_size), np.random.normal(3, 1, sample_size)


class CaseSameMeanDiffVar(TestCase):

    def sampling(self, sample_size):
        return np.random.normal(3, 1, sample_size), np.random.normal(3, 2, sample_size)


class CaseDiffMeanSameVar(TestCase):

    def sampling(self, sample_size):
        return np.random.normal(2, 1, sample_size), np.random.normal(3, 1, sample_size)


def simulation(test_method, test_case, n_iter=1000):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs, ys = test_case.sampling(test_case.sample_size)
        rejected[i] = test_method(xs, ys, test_case.alpha)
    return rejected


def main():
    rejected = simulation(t_test, CaseSameMeanSameVar())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(t_test, CaseSameMeanDiffVar())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(t_test, CaseDiffMeanSameVar())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()