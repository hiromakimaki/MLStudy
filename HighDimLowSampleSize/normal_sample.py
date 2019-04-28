"""
Summary:
    The script for checking the following fact:
        When x_1, x_2 ~ Normal(0, I_d) (i.i.d) and d -> infinity
    (I_d: d-dimensional identity matrix),
    then
        1) |x_1 - x_2| ~= sqrt(2 * d)
        2) angle(x_1, x_2) ~= pi / 2

Usage:
    python normal_sample.py

Requirements:
    numpy, pandas, matplotlib

Reference:
    chapter 2 of "高次元の統計学"(https://www.kyoritsu-pub.co.jp/bookdetail/9784320112636)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)


def get_sample(d, m = 0, v = 1):
    """
    Get two sample vectors x_1, x_2 identically from d-dimensional normal distribution.
    Then calculate and return two statistics:
        |x_1 - x_2|
        angle(x_1, x_2)
    """
    x = np.random.normal(m, v, d)
    y = np.random.normal(m, v, d)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    norm_x_y = np.linalg.norm(x - y)
    rad_xy = np.arccos(np.abs(np.dot(x, y)) / (norm_x * norm_y))
    return norm_x_y, rad_xy


def main():
    n_samples = 50
    dimensions = [1, 5, 10, 50, 100]
    norms = np.zeros(len(dimensions))
    radians = np.zeros(len(dimensions))
    sqrt_2d = np.zeros(len(dimensions))
    half_pi = np.pi / 2 * np.ones(len(dimensions))
    
    for i, d in enumerate(dimensions):
        sample_norms = np.zeros(n_samples)
        sample_radians = np.zeros(n_samples)
        for j in range(n_samples):
            norm_x_y, rad_xy = get_sample(d)
            sample_norms[j] = norm_x_y
            sample_radians[j] = rad_xy
        norms[i] = sample_norms.mean()
        radians[i] = sample_radians.mean()
        sqrt_2d[i] = np.sqrt(2*d)
    
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(dimensions, radians, label='mean angle', color='green')
    ax1.plot(dimensions, half_pi, label='pi/2', color='green', linestyle='dashed')
    ax1.legend(loc='center left')
    ax2.plot(dimensions, norms, label='mean norm', color='red')
    ax2.plot(dimensions, sqrt_2d, label='sqrt(2d)', color='red', linestyle='dashed')
    ax2.legend(loc='center right')
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    main()