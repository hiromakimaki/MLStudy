import numpy as np
from scipy import stats

np.random.seed(0)


def cochran_q_test(data, alpha):
    k = data.shape[1]
    A = data.sum(axis=0)
    B = data.sum(axis=1)
    Q = (k-1) * (k * (A**2).sum() - (A.sum())**2) / (k * B.sum() - (B**2).sum())
    upper = stats.chi2.ppf(q=1-alpha, df=k-1)
    if Q > upper:
        return 1
    return 0


def test():
    data = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,1],
        [0,1,1],
        [0,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])
    rejected = cochran_q_test(data, 0.05)
    assert rejected == 0


if __name__=='__main__':
    test()
