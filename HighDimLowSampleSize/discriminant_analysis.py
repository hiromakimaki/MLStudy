"""
Summary:
    (TODO: write this.)

Usage:
    (TODO: write this.)

Requirements:
    numpy, pandas, matplotlib

Reference:
    chapter 6 of "高次元の統計学"(https://www.kyoritsu-pub.co.jp/bookdetail/9784320112636)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(0)


class DiscriminantFunction:

    @staticmethod
    def func(class_0_data, class_1_data, target_data):
        """
        Memo:
            The value of `func` is negative -> class 0
            Otherwise                       -> class 1
        """
        raise NotImplementedError('Please implement this method.')

    @staticmethod
    def calc_cov(data):
        n = data.shape[1]
        P = np.eye(n) - np.ones((n, n)) / n
        XP = np.dot(data, P)
        return np.dot(XP, XP.T) / (n - 1)


class GeometricQuadraticFunction(DiscriminantFunction):
    @staticmethod
    def func(class_0_data, class_1_data, target_data):
        assert class_0_data.shape[0] == class_1_data.shape[0]
        assert class_0_data.shape[0] == target_data.shape[0]
        d = target_data.shape[0]
        n_0 = class_0_data.shape[1]
        n_1 = class_1_data.shape[1]

        mean_0 = class_0_data.mean(axis=1)
        mean_1 = class_1_data.mean(axis=1)

        cov_0 = DiscriminantFunction.calc_cov(class_0_data)
        cov_1 = DiscriminantFunction.calc_cov(class_1_data)

        quad_dist_0 = np.linalg.norm(target_data - mean_0.reshape((-1, 1)), axis=0)**2
        quad_dist_1 = np.linalg.norm(target_data - mean_1.reshape((-1, 1)), axis=0)**2

        Y = d * (quad_dist_0 / np.trace(cov_0) - quad_dist_1 / np.trace(cov_1))
        Y -= d * np.log(np.trace(cov_1) / np.trace(cov_0))
        Y -= d * (1 / n_0 - 1 / n_1)
        return Y


def sampling_from_same_mean_diff_cov(d, n_train, n_test):
    n_train_0 = n_train // 2
    n_train_1 = n_train - n_train_0
    n_test_0 = n_test // 2
    n_test_1 = n_test - n_test_0

    cov_0 = np.eye(d)
    cov_1 = 2 * np.eye(d)
    mean_0 = mean_1 = np.zeros(d)

    train_0 = np.random.multivariate_normal(mean_0, cov_0, n_train_0).T
    train_1 = np.random.multivariate_normal(mean_1, cov_1, n_train_1).T

    test_0 = np.random.multivariate_normal(mean_0, cov_0, n_test_0).T
    test_1 = np.random.multivariate_normal(mean_1, cov_1, n_test_1).T

    return train_0, train_1, test_0, test_1


def discriminant_analysis(discriminant_function):
    d = 2**10
    n_train, n_test = 50, 50
    train_0, train_1, test_0, test_1 = sampling_from_same_mean_diff_cov(d, n_train, n_test)
    func_0 = discriminant_function.func(train_0, train_1, test_0)
    func_1 = discriminant_function.func(train_0, train_1, test_1)
    func_values = np.concatenate([func_0, func_1])
    plt.scatter(np.arange(len(func_values)), func_values)
    plt.title('Values of {} on test data'.format(discriminant_function.__name__))
    plt.show()


def main():
    discriminant_analysis(GeometricQuadraticFunction)


if __name__=='__main__':
    main()