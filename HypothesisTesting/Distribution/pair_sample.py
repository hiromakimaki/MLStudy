import numpy as np
from scipy import stats


np.random.seed(0)


def wilcoxon_signed_rank_test(xs, ys, alpha):
    assert len(xs) == len(ys)
    zs = xs - ys
    nonzero_zs = zs[zs != 0]
    n = len(nonzero_zs)
    abs_nonzero_zs_ranks = stats.rankdata(np.abs(nonzero_zs))
    wx = abs_nonzero_zs_ranks[nonzero_zs > 0].sum()
    wy = abs_nonzero_zs_ranks[nonzero_zs < 0].sum()
    w = np.min([wx, wy])
    z = (w - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    lower, upper = stats.norm.ppf(q=[alpha/2, 1-alpha/2])
    if z < lower or upper < z:
        return 1
    return 0


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def sample_size(self):
        return 100

    def sampling_1(self, sample_size):
        pass

    def sampling_2(self, sample_size):
        pass

class SameNormalCase(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(0, 1, sample_size)

    def sampling_2(self, sample_size):
        return self.sampling_1(sample_size)


class DiffVarNormalCase(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(2, 4, sample_size)

    def sampling_2(self, sample_size):
        return np.random.normal(2, 8, sample_size)


class DiffMeanNormalCase(TestCase):

    def sampling_1(self, sample_size):
        return np.random.normal(0, 4, sample_size)

    def sampling_2(self, sample_size):
        return np.random.normal(2, 4, sample_size)


def simulation(test_case, test_method, n_iter=500):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = test_case.sampling_1(test_case.sample_size)
        ys = test_case.sampling_2(test_case.sample_size)
        rejected[i] = test_method(xs, ys, test_case.alpha)
    return rejected


def main():
    for test_case in [SameNormalCase(), DiffVarNormalCase(), DiffMeanNormalCase()]:
        rejected = simulation(test_case, wilcoxon_signed_rank_test)
        print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()
