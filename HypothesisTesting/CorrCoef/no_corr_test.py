from scipy import stats
import numpy as np

np.random.seed(0)


def pearson_corr_test(xs, ys, alpha):
    """
    Summary:
        Execute no correlation test with Pearson's correlation coefficient.
    Return value:
        0: The null hypothesis is NOT rejected.
        1: Rejected.
    """
    assert len(xs) == len(ys)
    n = len(xs)
    # r, pval = stats.pearsonr(xs, ys) # To calc a corr coef, this function is also useful.
    dx = xs - np.mean(xs)
    dy = ys - np.mean(ys)
    r = np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    t_value = np.sqrt(n-2) * np.abs(r) / np.sqrt(1 - r**2)
    lower, upper = stats.t.ppf(q=[alpha/2, 1-alpha/2], df=n-2)
    if t_value < lower or upper < t_value:
        return 1
    return 0


def spearman_rank_corr_test(xs, ys, alpha):
    assert len(xs) == len(ys)
    n = len(xs)
    x_ranks = stats.rankdata(xs)
    y_ranks = stats.rankdata(ys)
    d = x_ranks - y_ranks
    r = 1 - (6 * (d**2).sum()) / (n**3 - n) # Spearman's rank corr coef
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    lower, upper = stats.t.ppf(q=[alpha/2, 1-alpha/2], df=n-2)
    if t < lower or upper < t:
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


class CaseZeroCorr(TestCase):

    def sampling(self, sample_size):
        return np.random.normal(1, 4, sample_size), np.random.normal(2, 3, sample_size)


class CasePositiveCorr(TestCase):

    def sampling(self, sample_size):
        x = np.arange(sample_size)
        return x, x + np.random.normal(0, 1, sample_size)


class CaseSmallSampleSize(TestCase):

    @property
    def sample_size(self):
        return 8

    def sampling(self, sample_size):
        xs = np.arange(sample_size) + np.random.choice([-1, 0, 1], size=sample_size, p=[0.3, 0.4, 0.3])
        ys = np.ceil(xs / 2) + np.random.choice([-1, 0, 1], size=sample_size, p=[0.4, 0.2, 0.4])
        return xs, ys


def simulation(test_method, test_case, n_iter=1000):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs, ys = test_case.sampling(test_case.sample_size)
        rejected[i] = test_method(xs, ys, test_case.alpha)
    return rejected


def main():
    for test_case in [CaseZeroCorr(), CasePositiveCorr(), CaseSmallSampleSize()]:
        rejected = simulation(pearson_corr_test, test_case)
        print('Null hypothesis rejected ratio (Pearson): {} %'.format(100 * sum(rejected) / len(rejected)))
        rejected = simulation(spearman_rank_corr_test, test_case)
        print('Null hypothesis rejected ratio (Spearman): {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()