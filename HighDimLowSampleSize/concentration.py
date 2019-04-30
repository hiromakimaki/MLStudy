"""
Summary:
    Visualize eigenvector concentration of a dual sample covariance matrix
    on a spherical surface or coordinate axes.

Usage:
    python concentration.py -m ${mode}
        mode (optional):
            Specify either 'spherical_surface' or 'coordinate_axes'.
            Default is 'coordinate_axes'

Requirements:
    numpy, pandas, matplotlib

Reference:
    chapter 2 of "高次元の統計学"(https://www.kyoritsu-pub.co.jp/bookdetail/9784320112636)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

np.random.seed(0)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', dest='mode', choices=['spherical_surface', 'coordinate_axes'], default='coordinate_axes')
    return parser.parse_args()


def multivariate_t(d, mean, cov, degree_of_freedom, n):
    """
    Reference:
        https://qiita.com/kazetof/items/62f11b5b58e270e7dc77
    """
    V = np.random.gamma(shape=degree_of_freedom/2., scale=degree_of_freedom/2.,size=n)
    V = np.tile(V.reshape((n, 1)), d)
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    Y = mean + X / np.sqrt(V)
    return Y.T


def visualize_concentration(mode = 'coordinate_axes', d = 500, n_simulation_iters = 15):
    n = 3 # must not be changed
    mat_cov = np.eye(d)
    arr_mean = np.zeros(d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _ in range(n_simulation_iters):
        # Sampling from t distribution
        if mode == 'coordinate_axes':
            degree_of_freedom = 5
            X = multivariate_t(d, arr_mean, mat_cov, degree_of_freedom, n)
        elif mode == 'spherical_surface':
            X = np.random.multivariate_normal(arr_mean, mat_cov, n).T
        else:
            raise Exception('Invalid `mode` is specified: {}'.format(mode))
        # Calculating dual covariance matrix
        X_centered = X - np.dot(arr_mean.reshape((d, 1)), np.ones((1, n)))
        mat_dual_cov = np.dot(X_centered.T, X_centered) / n
        # Eigenvalue decomposition
        _, eigvecs = np.linalg.eigh(mat_dual_cov)
        # Scaling eigenvectors
        wvecs = n * np.dot(mat_dual_cov, eigvecs) / np.trace(mat_cov)
        # Visualizing
        ax.scatter(wvecs[0,:], wvecs[1,:], wvecs[2,:])
        ax.scatter(-wvecs[0,:], -wvecs[1,:], -wvecs[2,:])
    plt.title('Eigenvector map')
    plt.show()


def main(args):
    visualize_concentration(mode=args.mode)


if __name__ == '__main__':
    main(parse_args())