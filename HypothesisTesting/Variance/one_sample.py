import numpy as np
from scipy import stats

np.random.seed(0)


def chisquare_test(xs, variance, alpha):
    """
    Summary:
        Execute one-sample two-sided chisquare-test.
    Return value:
        0: The null hypothesis is NOT rejected.
        1: Rejected.
    """
    n = len(xs)
    x = n * np.var(xs) / variance
    lower, upper = stats.chi2.ppf(q=[alpha/2, 1-alpha/2], df=n-1)
    if x < lower or upper < x:
        return 1
    return 0


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def sample_size(self):
        return 50

    @property
    def variance(self):
        pass

    def sampling(self, sample_size):
        pass

class CaseToBeAccepted(TestCase):

    @property
    def variance(self):
        return 3**2

    def sampling(self, sample_size):
        return np.random.normal(1, 3, sample_size)


class CaseToBeRejected(TestCase):

    @property
    def variance(self):
        return 3**2

    def sampling(self, sample_size):
        return np.random.normal(1, 2, sample_size)


def simulation(test_case, n_iter=1000):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = test_case.sampling(test_case.sample_size)
        rejected[i] = chisquare_test(xs, test_case.variance, test_case.alpha)
    return rejected


def main():
    for test_case in [CaseToBeAccepted(), CaseToBeRejected()]:
        rejected = simulation(test_case)
        print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()