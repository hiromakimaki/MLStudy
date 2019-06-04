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


def kendall_rank_corr_test(xs, ys, alpha):
    assert len(xs) == len(ys)
    n = len(xs)
    mat_xs = np.repeat(xs.reshape((1, n)), n, axis=0)
    diff_mat_xs = mat_xs - mat_xs.T
    mat_ys = np.repeat(ys.reshape((1, n)), n, axis=0)
    diff_mat_ys = mat_ys - mat_ys.T
    mat_sign = np.sign(diff_mat_xs * diff_mat_ys)
    n_same_sign = np.sum(mat_sign > 0) / 2
    n_diff_sign = np.sum(mat_sign < 0) / 2
    tau = (n_same_sign - n_diff_sign) / (n * (n + 1) / 2) # Kendall's rank corr coef
    v = 2 * (2 * n + 5) / (9 * n * (n - 1))
    t = tau / np.sqrt(v)
    lower, upper = stats.norm.ppf(q=[alpha/2, 1-alpha/2])
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
        return 10

    def sampling(self, sample_size):
        xs = np.arange(sample_size) + np.random.choice([-1, 0, 1], size=sample_size, p=[0.3, 0.4, 0.3])
        ys = np.ceil(xs / 2) + np.random.choice([-1, 0, 1], size=sample_size, p=[0.4, 0.2, 0.4])
        return xs, ys


class CaseSmallSampleSizeNoCorr(TestCase):

    @property
    def sample_size(self):
        return 10

    def sampling(self, sample_size):
        choices = np.arange(-10, 10)
        d = len(choices)
        prob = np.ones(d) / d
        xs = np.random.choice(choices, size=sample_size, p=prob)
        ys = np.random.choice(choices, size=sample_size, p=prob)
        return xs, ys


def simulation(test_method, test_case, n_iter=1000):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs, ys = test_case.sampling(test_case.sample_size)
        rejected[i] = test_method(xs, ys, test_case.alpha)
    return rejected


def main():
    for test_case in [CaseZeroCorr(), CasePositiveCorr(), CaseSmallSampleSize(), CaseSmallSampleSizeNoCorr()]:
        rejected = simulation(pearson_corr_test, test_case)
        print('Null hypothesis rejected ratio (Pearson): {} %'.format(100 * sum(rejected) / len(rejected)))
        rejected = simulation(spearman_rank_corr_test, test_case)
        print('Null hypothesis rejected ratio (Spearman): {} %'.format(100 * sum(rejected) / len(rejected)))
        rejected = simulation(kendall_rank_corr_test, test_case)
        print('Null hypothesis rejected ratio (Kendall): {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()