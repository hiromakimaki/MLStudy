"""
Summary:
    (TODO: Write this!)

Usage:
    python eigen_value_estimation.py

Requirements:
    scipy, numpy, pandas, matplotlib

Reference:
    chapter 3 of "高次元の統計学"(https://www.kyoritsu-pub.co.jp/bookdetail/9784320112636)
"""

from scipy.sparse.linalg import eigsh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)


def get_sample_top_k_eigenvalues(mat_cov, n, k):
    d = mat_cov.shape[0]
    X = np.random.multivariate_normal(np.zeros(d), mat_cov, n).T
    P = np.eye(n) - np.ones((n,n)) / n
    XP = np.dot(X, P)
    mat_dual_cov = np.dot(XP, XP.T) / (n - 1)
    eigvals, _ = eigsh(mat_dual_cov, k=k)
    eigvals = - (np.sort(- eigvals)) # descending order
    return eigvals


def generate_mat_cov(d, eig_1, eig_2):
    assert d > 2
    mat_cov = np.eye(d)
    mat_cov[0, 0] = eig_1
    mat_cov[1, 1] = eig_2
    return mat_cov


def main():
    n_simulation_iters = 10
    ds = 2**np.array([6, 7, 8, 9, 10])
    eig_1_ratios = np.zeros(ds.shape[0])
    eig_2_ratios = np.zeros(ds.shape[0])
    for i, d in enumerate(ds):
        # Constants
        n = np.ceil(d**(1/3)).astype(np.int)
        eig_1 = d**(2/3)
        eig_2 = d**(1/2)
        k = 2
        # Simulation
        mat_cov = generate_mat_cov(d, eig_1, eig_2)
        est_eigvals = np.zeros((n_simulation_iters, k))
        for j in range(n_simulation_iters):
            est_eigvals[j,:] = get_sample_top_k_eigenvalues(mat_cov, n, k)
        est_eigvals = est_eigvals.mean(axis=0)
        eig_1_ratios[i] = est_eigvals[0] / eig_1
        eig_2_ratios[i] = est_eigvals[1] / eig_2
        print('Fin. {}-th loop'.format(i))
    plt.plot(ds, eig_1_ratios, label='1st eigval ratio')
    plt.plot(ds, eig_2_ratios, label='2nd eigval ratio')
    plt.xlabel('d')
    plt.ylabel('ratio of estimated eigenvalue and true eigenvalue')
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()