import numpy as np
from scipy import stats

np.random.seed(0)


def t_test(xs, mu, alpha):
    """
    Summary:
        Execute one-sample two-sided t-test.
    Return value:
        0: The null hypothesis is NOT rejected.
        1: Rejected.
    """
    n = len(xs)
    t_value = np.sqrt(n) * (np.mean(xs) - mu) / np.std(xs, ddof=1)
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

    @property
    def mu(self):
        pass

    def sampling(self, sample_size):
        pass

class CaseToBeAccepted(TestCase):

    @property
    def mu(self):
        return 3

    def sampling(self, sample_size):
        return np.random.normal(3, 1, sample_size)


class CaseToBeRejected(TestCase):

    @property
    def mu(self):
        return 3

    def sampling(self, sample_size):
        return np.random.normal(2.5, 1, sample_size)


class CaseExceptional(TestCase):

    @property
    def mu(self):
        return 3

    def sampling(self, sample_size):
        return 3 + np.random.standard_cauchy(sample_size)


def simulation(test_case, n_iter=1000):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = test_case.sampling(test_case.sample_size)
        rejected[i] = t_test(xs, test_case.mu, test_case.alpha)
    return rejected


def main():
    rejected = simulation(CaseToBeAccepted())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(CaseToBeRejected())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(CaseExceptional())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()