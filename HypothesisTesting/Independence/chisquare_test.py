from scipy import stats
import numpy as np

np.random.seed(0)


def chisquare_test(data, alpha):
    """
    Summary:
        Execute chisquare test for independence.
        The null hypothesis is that X, Y are independent.
    Return value:
        0: The null hypothesis is NOT rejected.
        1: Rejected.
    """
    dim_x, dim_y = data.shape
    n = data.sum()
    x = 0
    for i in range(dim_x):
        for j in range(dim_y):
            expected = data[i,:].sum() * data[:,j].sum() / n
            x += ((data[i,j] - expected)**2) / expected
    upper = stats.chi2.ppf(q=1-alpha, df=(dim_x-1)*(dim_y-1))
    if x > upper:
        return 1
    return 0


class TestCase:
    @property
    def alpha(self):
        return 0.05

    @property
    def sample_size(self):
        return 100

    def sampling(self, sample_size):
        pass

class CaseToBeAccepted(TestCase):

    def sampling(self, sample_size):
        dim_x = 2
        prob_x = np.array([i + 1 for i in range(dim_x)])
        prob_x = prob_x / prob_x.sum()
        dim_y = 3
        prob_y = np.array([i + 2 for i in range(dim_y)])
        prob_y = prob_y / prob_y.sum()
        data = np.zeros((dim_x, dim_y))
        for _ in range(sample_size):
            i = np.random.choice(list(range(dim_x)), size = 1, p=prob_x)[0]
            j = np.random.choice(list(range(dim_y)), size = 1, p=prob_y)[0]
            data[i, j] = data[i, j] + 1
        return data


class CaseToBeRejected(TestCase):

    def sampling(self, sample_size):
        dim_x = dim_y = 2
        choices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        prob_xy = np.array([7, 3, 3, 7])
        prob_xy = prob_xy / prob_xy.sum()
        data = np.zeros((dim_x, dim_y))
        for _ in range(sample_size):
            index = np.random.choice(list(range(dim_x*dim_y)), size = 1, p=prob_xy)[0]
            i, j = choices[index]
            data[i, j] = data[i, j] + 1
        return data


def simulation(test_case, n_iter=500):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        data = test_case.sampling(test_case.sample_size)
        rejected[i] = chisquare_test(data, test_case.alpha)
    return rejected


def main():
    for test_case in [CaseToBeAccepted(), CaseToBeRejected()]:
        rejected = simulation(test_case)
        print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


if __name__=='__main__':
    main()