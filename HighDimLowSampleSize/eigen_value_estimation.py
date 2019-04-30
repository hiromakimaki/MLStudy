"""
Summary:
    Compare estimation methods of eigenvalues of covariance matrix with high dimension.
    The methods are:
        1) Simple method
            Calculate the sample covariance matrix, then calculate its eigenvalues.
            -> The estimated values tend to be overestimated.
        2) Noise reduction method
            (See the reference)

Usage:
    python eigen_value_estimation.py

Requirements:
    scipy, numpy, pandas, matplotlib

Reference:
    chapter 4 of "高次元の統計学"(https://www.kyoritsu-pub.co.jp/bookdetail/9784320112636)
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
    # Noise reduction method
    denominators = n - 1 - np.arange(1, k+1)
    noise_reduced_eigvals = eigvals - (np.trace(mat_dual_cov) - eigvals.cumsum()) / denominators
    return eigvals, noise_reduced_eigvals


def generate_mat_cov(d, eig_1, eig_2):
    assert d > 2
    mat_cov = np.eye(d)
    mat_cov[0, 0] = eig_1
    mat_cov[1, 1] = eig_2
    return mat_cov


def main():
    n_simulation_iters = 20
    ts = np.array([5, 6, 7, 8, 9])
    ds = 2**ts
    eig_1_ratios = np.zeros(ds.shape[0])
    eig_2_ratios = np.zeros(ds.shape[0])
    noise_reduced_eig_1_ratios = np.zeros(ds.shape[0])
    noise_reduced_eig_2_ratios = np.zeros(ds.shape[0])
    for i, d in enumerate(ds):
        # Constants
        n = np.ceil(d**(1/3)).astype(np.int)
        eig_1 = d**(2/3)
        eig_2 = d**(1/2)
        k = 2
        # Simulation
        mat_cov = generate_mat_cov(d, eig_1, eig_2)
        est_eigvals = np.zeros((n_simulation_iters, k))
        noise_reduced_eigvals = np.zeros((n_simulation_iters, k))
        for j in range(n_simulation_iters):
            tmp_est_eigvals, tmp_noise_reduced_eigvals = get_sample_top_k_eigenvalues(mat_cov, n, k)
            est_eigvals[j,:] = tmp_est_eigvals
            noise_reduced_eigvals[j,:] = tmp_noise_reduced_eigvals
        est_eigvals = est_eigvals.mean(axis=0)
        noise_reduced_eigvals = noise_reduced_eigvals.mean(axis=0)
        eig_1_ratios[i] = est_eigvals[0] / eig_1
        eig_2_ratios[i] = est_eigvals[1] / eig_2
        noise_reduced_eig_1_ratios[i] = noise_reduced_eigvals[0] / eig_1
        noise_reduced_eig_2_ratios[i] = noise_reduced_eigvals[1] / eig_2
        print('Fin. {}-th loop'.format(i+1))
    plt.plot(ts, eig_1_ratios, label=r'$\hat \lambda_{(1)} / \lambda_{(1)}$', marker='^', color='blue')
    plt.plot(ts, eig_2_ratios, label=r'$\hat \lambda_{(2)} / \lambda_{(2)}$', marker='^', color='blue', linestyle='dashed')
    plt.plot(ts, noise_reduced_eig_1_ratios, label=r'$\tilde \lambda_{(1)} / \lambda_{(1)}$ (NR)', marker='o', color='green')
    plt.plot(ts, noise_reduced_eig_2_ratios, label=r'$\tilde \lambda_{(2)} / \lambda_{(2)}$ (NR)', marker='o', color='green', linestyle='dashed')
    plt.xlabel(r'$\log_2 d$')
    plt.ylabel('ratio')
    plt.title('ratio of estimated eigenvalue and true eigenvalue')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()