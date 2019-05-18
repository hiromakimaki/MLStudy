import numpy as np
from scipy import stats

np.random.seed(0)


def t_test(xs, ys, alpha):
    """
    Summary:
        Execute two-sample two-sided t-test.
        It is supposed that the values of (unknown) variances of the two samples are same.
    Return value:
        0: The null hypothesis is NOT rejected.
        1: Rejected.
    """
    nx = len(xs)
    ny = len(ys)
    est_var = ((nx - 1) * (np.std(xs, ddof=1)**2) + (ny - 1) * (np.std(xs, ddof=1)**2)) / (nx + ny - 2)
    t_value = (np.mean(xs) - np.mean(ys)) / np.sqrt((1/nx + 1/ny) * est_var)
    lower, upper = stats.t.ppf(q=[alpha/2, 1-alpha/2], df=nx+ny-2)
    if t_value < lower or upper < t_value:
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

    def sampling_1(self, sample_size):
        pass

    def sampling_2(self, sample_size):
        pass


class CaseSameMeanSameVar(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(3, 1, sample_size)

    def sampling_2(self, sample_size):
        return np.random.normal(3, 1, sample_size)


class CaseSameMeanDiffVar(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(3, 1, sample_size)

    def sampling_2(self, sample_size):
        return np.random.normal(3, 2, sample_size)


class CaseDiffMeanSameVar(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(2, 1, sample_size)

    def sampling_2(self, sample_size):
        return np.random.normal(3, 1, sample_size)


def simulation(test_case, n_iter=1000):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = test_case.sampling_1(test_case.sample_size_1)
        ys = test_case.sampling_2(test_case.sample_size_2)
        rejected[i] = t_test(xs, ys, test_case.alpha)
    return rejected


def main():
    rejected = simulation(CaseSameMeanSameVar())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(CaseSameMeanDiffVar())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))
    rejected = simulation(CaseDiffMeanSameVar())
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()