"""
References:
    Gibbons, J. D., and S. Chakraborti. "Nonparametric Statistical Inference"
"""
import numpy as np
from scipy import stats

np.random.seed(0)


def birnbaum_tingey_test(xs, alpha):
    """
    See Corollary 3.5 in chapter 4 in the reference book.
    """
    n = len(xs)
    ordered_xs = np.sort(xs)
    fn = np.ones(n).cumsum() / n
    d_plus = np.max(fn - stats.norm.cdf(ordered_xs)) # compare to standard normal
    d_plus = np.max([d_plus, 0])
    v = 4 * n * (d_plus**2)
    upper = stats.chi2.ppf(q=1-alpha, df=2) # equal to exponential distribution with mean = 2
    if v > upper:
        return 1
    return 0


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def sample_size(self):
        return 200

    def sampling(self, sample_size):
        pass


class NormalCase(TestCase):

    def sampling(self, sample_size):
        return np.random.normal(0, 1, sample_size)


class NormalNotStandardCase(TestCase):

    def sampling(self, sample_size):
        return np.random.normal(5, 8, sample_size)


class LaplaceCase(TestCase):

    def sampling(self, sample_size):
        return np.random.laplace(0, 1, sample_size)


class CauchyCase(TestCase):

    def sampling(self, sample_size):
        return np.random.standard_cauchy(sample_size)


def simulation(test_case, n_iter=500):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        data = test_case.sampling(test_case.sample_size)
        rejected[i] = birnbaum_tingey_test(data, test_case.alpha)
    return rejected


def main():
    rejected = simulation(NormalCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(NormalNotStandardCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(LaplaceCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(CauchyCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()