"""
Summary:
    Compare mean squared losses of 2 estimators:
    the least squares estimator and the James-Stein estimator.

    Memo for James-Stein estimator:
        Y ~ N(theta, I) (dim of Y is d (>= 3)).
        Estimate the mean parameter `theta` from only one sample `y`.
        ->
        Least squares estimator: y
        James-Stein estimator: (1 - (d - 2) / |y|^2) * y
        ->
        Mean squared losses for 2 estimators are:
            Least squares estimator: d
            James-Stein estimator: d - (d - 2)^2 * E[1 / (d - 2 + 2 * K)]
                (K ~ Poisson(|theta|^2 / 2)).
        The loss of the James-Stein estimator is SMALLER THAN that of the least squares estimator.

Usage:
    python james_stein.py

Requirements:
    numpy, pandas, matplotlib

Reference:
    James, W.; Stein, C. (1961), "Estimation with Quadratic Loss" (https://projecteuclid.org/euclid.bsmsp/1200512173)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)


def get_param(base_num, d):
    if base_num <= 0:
        return np.zeros(d)
    return  float(base_num)**np.arange(d)


def simulation(base_num, dims, n_simulations):
    ls_mse = np.zeros(len(dims)) # Least squares estimator
    js_mse = np.zeros(len(dims)) # James-Stein estimator
    ls_true_mse = np.zeros(len(dims)) # d
    js_true_mse = np.zeros(len(dims)) # d - (d - 2)^2 * E[1 / (d - 2 + 2 * K)], (K ~ Poisson(|theta|^2 / 2))
    for i, d in enumerate(dims):
        theta = get_param(base_num, d)
        ls_true_mse[i] = d
        js_true_mse[i] = d - ((d - 2)**2) * np.mean(1 / (d - 2 + 2 * np.random.poisson((np.linalg.norm(theta)**2) / 2, 10000)))
        ls_squared_error = np.zeros(n_simulations)
        js_squared_error = np.zeros(n_simulations)
        for n in range(n_simulations):
            y = np.random.multivariate_normal(theta, np.eye(d), 1)[0]
            ls_theta = y.copy()
            ls_squared_error[n] = np.linalg.norm(theta - ls_theta)**2
            js_theta = (1 - (d - 2) / (np.linalg.norm(y)**2)) * y
            js_squared_error[n] = np.linalg.norm(theta - js_theta)**2
        ls_mse[i] = ls_squared_error.mean()
        js_mse[i] = js_squared_error.mean()
    return ls_mse, js_mse, ls_true_mse, js_true_mse


def main():
    dims = [3, 10, 20, 30]
    n_simulations = 100
    for base_num in [0, 0.9, 1, 1.1]:
        ls_mse, js_mse, ls_true_mse, js_true_mse = simulation(base_num, dims, n_simulations)
        plt.plot(dims, np.zeros(len(dims)), linestyle='--', c='black')
        plt.plot(dims, ls_mse, label='Least squares (Simulated)', linestyle='-', c='blue')
        plt.plot(dims, js_mse, label='James-Stein (Simulated)', linestyle='-', c='orange')
        plt.plot(dims, ls_true_mse, label='Least squares (True)', linestyle='--', c='blue')
        plt.plot(dims, js_true_mse, label='James-Stein (True)', linestyle='--', c='orange')
        plt.legend(loc='upper left')
        plt.title('MSE in the case: base_num = {}'.format(base_num))
        plt.show()


if __name__=='__main__':
    main()