from scipy import stats
import numpy as np

np.random.seed(0)


def sample_data():
    return np.array([84, 81]), np.array([69, 73, 65, 74]), np.array([77, 68, 70])


def kruskal_wallis_test(alpha, *x):
    n_group = len(x)
    x_rank = stats.rankdata(np.concatenate(x))
    n = len(x_rank)
    n_kth_group = list()
    rank_sum = list()
    index = 0
    for k in range(n_group):
        n_kth = len(x[k])
        n_kth_group.append(n_kth)
        rank_sum.append(x_rank[index:(index+n_kth)].sum())
        index += n_kth
    n_kth_group = np.array(n_kth_group)
    rank_sum = np.array(rank_sum)
    t = 12 * ((rank_sum**2) / n_kth_group).sum() / (n * (n + 1)) - 3 * (n + 1)
    upper = stats.chi2.ppf(q=1-alpha, df=n_group-1)
    if upper < t:
        return 1
    return 0


def main():
    # Expected: not rejected
    assert 0 == kruskal_wallis_test(0.05, *sample_data())
    print('OK')


if __name__=='__main__':
    main()