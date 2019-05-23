import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def same_sample_size_case():
    sample_size = 100
    x = np.random.normal(0, 1, sample_size)
    x = np.sort(x)
    y = np.random.normal(3, 2, sample_size)
    y = np.sort(y)
    z = np.random.standard_cauchy(sample_size)
    z = np.sort(z)
    # Case 1
    plt.scatter(x, y)
    plt.show()
    # Case 2
    plt.scatter(x, z)
    plt.show()


def main():
    same_sample_size_case()


if __name__=='__main__':
    main()