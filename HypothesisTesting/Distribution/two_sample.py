"""
References:
    Gibbons, J. D., and S. Chakraborti. "Nonparametric Statistical Inference"
"""
import numpy as np
from scipy import stats

np.random.seed(0)


def birnbaum_tingey_test(xs, ys, alpha):
    """
    This name derives from that of the one-sample case.
    See section 6.3 in the reference book.
    """
    nx = len(xs)
    ny = len(ys)
    ordered_xs = np.sort(xs)
    ordered_ys = np.sort(ys)
    ordered_xys = np.sort(np.concatenate([xs, ys]))
    emp_dist_x = np.frompyfunc(lambda z: (z >= ordered_xs).sum() / nx, 1, 1)
    emp_dist_y = np.frompyfunc(lambda z: (z >= ordered_ys).sum() / ny, 1, 1)
    fx = emp_dist_x(ordered_xys)
    fy = emp_dist_y(ordered_xys)

    d_plus = np.max(fx - fy)
    v = 4 * (nx * ny / (nx + ny)) * (d_plus**2)

    upper = stats.chi2.ppf(q=1-alpha, df=2) # equal to exponential distribution with mean = 2
    if v > upper:
        return 1
    return 0


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def sample_size_1(self):
        return 150

    @property
    def sample_size_2(self):
        return 170

    def sampling_1(self, sample_size):
        pass

    def sampling_2(self, sample_size):
        pass


class SameNormalCase(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(0, 1, sample_size)

    def sampling_2(self, sample_size):
        return self.sampling_1(sample_size)


class DiffNormalCase(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(2, 4, sample_size)

    def sampling_2(self, sample_size):
        return np.random.normal(2, 8, sample_size)


class NormalLaplaceCase(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(0, 1, sample_size)

    def sampling_2(self, sample_size):
        return np.random.laplace(0, 1, sample_size)


class NormalCauchyCase(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(0, 1, sample_size)

    def sampling_2(self, sample_size):
        return np.random.standard_cauchy(sample_size)


def simulation(test_case, n_iter=500):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = test_case.sampling_1(test_case.sample_size_1)
        ys = test_case.sampling_2(test_case.sample_size_2)
        rejected[i] = birnbaum_tingey_test(xs, ys, test_case.alpha)
    return rejected


def main():
    rejected = simulation(SameNormalCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(DiffNormalCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(NormalLaplaceCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(NormalCauchyCase())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()