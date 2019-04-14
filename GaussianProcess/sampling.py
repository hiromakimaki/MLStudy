"""
Summary:
  Sampling from Gaussian process and plotting them.

Usage:
  python sampling.py

Requirements:
  numpy, pandas, matplotlib

Reference:
  Chapter 3(-3.3) of "ガウス過程と機械学習（機械学習プロフェッショナルシリーズ）"(https://www.kspub.co.jp/book/detail/1529267.html)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class KernelCalculator:
    def kernel(self, arr_x):
        raise NotImplementedError('Please implement this method.')


class GaussianKernel(KernelCalculator):
    def kernel(self, arr_x):
        dim_x = arr_x.shape[0]
        mat_x = np.dot(np.diag(arr_x), np.ones((dim_x, dim_x)))
        return np.exp(- (mat_x - mat_x.T)**2)


class ExponentialKernel(KernelCalculator):
    def kernel(self, arr_x):
        dim_x = arr_x.shape[0]
        mat_x = np.dot(np.diag(arr_x), np.ones((dim_x, dim_x)))
        return np.exp(- np.abs(mat_x - mat_x.T))


class LinearKernel(KernelCalculator):
    def kernel(self, arr_x):
        dim_x = arr_x.shape[0]
        return np.dot(arr_x.reshape(dim_x, 1), arr_x.reshape(1, dim_x)) + 1


class PeriodicKernel(KernelCalculator):
    def kernel(self, arr_x):
        dim_x = arr_x.shape[0]
        mat_x = np.dot(np.diag(arr_x), np.ones((dim_x, dim_x)))
        return np.exp(np.cos(np.abs(mat_x - mat_x.T)))


class GPSampler:
    def __init__(self, kernel_calculator):
        self._kc = kernel_calculator
    
    def set_kernel_calculator(self, kernel_calculator):
        self._kc = kernel_calculator
    
    def sample(self, arr_x):
        # Check input
        assert len(arr_x.shape) == 1
        dim_x = arr_x.shape[0]
        # Kernel
        kernel = self._kc.kernel(arr_x)
        # Sampling
        sample = np.random.multivariate_normal(mean=np.zeros(dim_x), cov=kernel)
        return sample
    
    def kernel_name(self):
        return self._kc.__class__.__name__


def main():
    num_samples = 5
    kernels = [GaussianKernel(), ExponentialKernel(), LinearKernel(), PeriodicKernel()]
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    arr_x = np.arange(-7, 7, 0.1)
    for j in [0, 1]:
        for i in [0, 1]:
            kernel_index = i + 2 * j
            sampler = GPSampler(kernels[kernel_index])
            plt.subplot(2, 2, (1 + kernel_index))
            for _ in range(num_samples):
                s = sampler.sample(arr_x)
                plt.plot(arr_x, s)
            plt.title('Samples from {}'.format(sampler.kernel_name()))
    plt.show()


if __name__=='__main__':
    main()

