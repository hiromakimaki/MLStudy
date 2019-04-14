"""
Summary:
  Execute Gaussian process regression for dummy data.
  The observation noise is considered in this model.

Usage:
  python regression.py

Requirements:
  scipy, numpy, pandas, matplotlib

Reference:
  Chapter 3(3.4-3.5) of "ガウス過程と機械学習（機械学習プロフェッショナルシリーズ）"(https://www.kspub.co.jp/book/detail/1529267.html)
"""

import pandas as pd
import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt

np.random.seed(0)

class KernelCalculator:
    @property
    def dim_params(self):
        raise NotImplementedError('Please implement this method.')

    def kernel(self, arr_x, params):
        raise NotImplementedError('Please implement this method.')

    def grad_kernel(self, arr_x, params):
        raise NotImplementedError('Please implement this method.')


class GaussianKernel(KernelCalculator):
    @property
    def dim_params(self):
        return 2 # length of `params`

    def kernel(self, arr_x, params):
        assert self.dim_params == params.shape[0]
        # params
        exp_coef = np.exp(params[0])
        scale = np.exp(params[1])
        # calc kernel
        dim_x = arr_x.shape[0]
        mat_x = np.dot(np.diag(arr_x), np.ones((dim_x, dim_x)))
        return exp_coef * np.exp(- ((mat_x - mat_x.T)**2) / scale)

    def grad_kernel(self, arr_x, params):
        assert self.dim_params == params.shape[0]
        # params
        #exp_coef = np.exp(params[0]) # not need in this function
        scale = np.exp(params[1])
        # calc kernel gradients
        dim_x = arr_x.shape[0]
        mat_x = np.dot(np.diag(arr_x), np.ones((dim_x, dim_x)))
        grad_kernel_1 = self.kernel(arr_x, params)
        grad_kernel_1 = grad_kernel_1.reshape((1, dim_x, dim_x))
        grad_kernel_2 = ((mat_x - mat_x.T)**2) / scale * self.kernel(arr_x, params)
        grad_kernel_2 = grad_kernel_2.reshape((1, dim_x, dim_x))
        return np.concatenate([grad_kernel_1, grad_kernel_2])


class GPRegressor:
    """
    Gaussian process regressor class with Gaussian noise.
    """
    def __init__(self, kernel_calculator):
        self._kc = kernel_calculator
        self._params = None
        self._train_x = None
        self._train_y = None

    def fit(self, arr_x, arr_y):
        def neg_loglik(params):
            return - self._loglik(self._kc.kernel, arr_x, arr_y, params)
        init_params = np.zeros(1 + self._kc.dim_params) # first dim is for the observation noise (Gaussian)
        res = opt.minimize(neg_loglik, init_params, method='Powell')
        print(res)
        self._params = res.x
        self._train_x = arr_x
        self._train_y = arr_y

    def _check_members_for_prediction(self):
        assert self._params is not None
        assert self._train_x is not None
        assert self._train_y is not None

    def predict(self, test_x):
        self._check_members_for_prediction()
        # Constant
        dim_train_x = self._train_x.shape[0]
        # Calc Kernel
        all_kernel = self._all_kernel_with_gaussian_noise(self._kc.kernel, self._train_x, test_x, self._params)        
        inv_k = np.linalg.inv(all_kernel[:dim_train_x, :dim_train_x])
        k_star = all_kernel[:dim_train_x, dim_train_x:]
        k_star_star = all_kernel[dim_train_x:, dim_train_x:]
        # Regression
        mu = np.dot(np.dot(k_star.T, inv_k), self._train_y)
        var = np.diag(k_star_star - np.dot(np.dot(k_star.T, inv_k), k_star))
        return mu, var

    @staticmethod
    def _split_params(params):
        noise_scale = np.exp(params[0])
        params_for_kernel = params[1:]
        return noise_scale, params_for_kernel

    @staticmethod
    def _all_kernel_with_gaussian_noise(kernel_func, arr_train_x, arr_test_x, params):
        noise_scale, params_for_kernel = GPRegressor._split_params(params)
        dim_train_x = arr_train_x.shape[0]
        all_x = np.concatenate([arr_train_x, arr_test_x])
        kernel = kernel_func(all_x, params_for_kernel)
        # Add Gaussian noise term only for train data
        kernel[:dim_train_x, :dim_train_x] = kernel[:dim_train_x, :dim_train_x] + np.eye(dim_train_x) * noise_scale
        return kernel

    @staticmethod
    def _kernel_with_gaussian_noise(kernel_func, arr_train_x, params):
        noise_scale, params_for_kernel = GPRegressor._split_params(params)
        dim_train_x = arr_train_x.shape[0]
        kernel = kernel_func(arr_train_x, params_for_kernel)
        kernel += np.eye(dim_train_x) * noise_scale # Add Gaussian noise term
        return kernel

    @staticmethod
    def _loglik(kernel_func, arr_train_x, arr_train_y, params):
        assert arr_train_x.shape[0] == arr_train_y.shape[0]
        dim_train_x = arr_train_x.shape[0]
        kernel = GPRegressor._kernel_with_gaussian_noise(kernel_func, arr_train_x, params)
        inv_kernel = np.linalg.inv(kernel)
        # log likelihood
        ll = - np.log(np.linalg.det(kernel)) / 2.0
        ll += - np.dot(arr_train_y, np.dot(inv_kernel, arr_train_y)) / 2.0
        ll += - dim_train_x * np.log(2 * np.pi) / 2.0
        return ll


def main():
    # Sampling data
    train_x = np.concatenate([np.arange(-5, 0, 0.5), np.arange(2, 5, 0.3)])
    train_y = 3.2*np.sin(train_x) + np.random.normal(0, 1, train_x.shape[0])
    # Estimate params
    regressor = GPRegressor(GaussianKernel())
    regressor.fit(train_x, train_y)
    # Prediction
    test_x = np.arange(-6, 6, 0.1)
    mu, var = regressor.predict(test_x)
    # Plotting
    sd = np.sqrt(var)
    plt.scatter(train_x, train_y, marker='*')
    plt.plot(test_x, mu)
    plt.fill_between(test_x, mu + 2*sd , mu - 2*sd, facecolor='g', alpha=0.5)
    plt.show()


if __name__=='__main__':
    main()

