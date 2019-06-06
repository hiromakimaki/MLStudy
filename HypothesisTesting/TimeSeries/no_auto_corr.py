import numpy as np


def auto_corr(xs, m):
    n = len(xs)
    assert n > m
    mu = np.mean(xs)
    auto_corrs = np.zeros(m+1)
    for lag in range(m+1):
        ys = xs[:n-lag]
        zs = xs[lag:]
        auto_corrs[lag] = np.mean((ys - mu) * (zs  - mu))
    auto_corrs = auto_corrs / auto_corrs[0]
    return auto_corrs


def main():
    xs = np.random.normal(0, 5, 20)
    print(auto_corr(xs, 10))
    pass

if __name__=='__main__':
    main()