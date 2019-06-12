import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


np.random.seed(0)


def ols_estimator(xs):
    return (xs[1:] * xs[:-1]).sum() / (xs**2).sum()


def generate_ar1_data(n, coef):
    xs = np.zeros(n)
    for i in range(1, n):
        xs[i] = coef * xs[i-1] + np.random.normal(0, 1)
    return xs


def dickey_fuller_test(xs, alpha):
    # TODO: Implement this
    return 0


def simulation(n_iter=500):
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = np.random.normal(0, 1, 100)
        rejected[i] = dickey_fuller_test(xs, 0.05)
    return rejected


def main():
    rejected = simulation()
    print('Null hypothesis rejected ratio: {} %'.format(100 * sum(rejected) / len(rejected)))


def check_distribution(n_iter=500):
    tau_x = np.zeros(n_iter)
    tau_y = np.zeros(n_iter)
    tau_z = np.zeros(n_iter)
    rho_x = 0
    rho_y = 0.5
    rho_z = 1
    n = 500
    for i in range(n_iter):
        xs = generate_ar1_data(n, rho_x) # not unit root process
        tau_x[i] = np.sqrt(n) * (ols_estimator(xs) - rho_x)
        ys = generate_ar1_data(n, rho_y) # not unit root process
        tau_y[i] = np.sqrt(n) * (ols_estimator(ys) - rho_y)
        zs = generate_ar1_data(n, rho_y) # unit root process
        tau_z[i] = np.sqrt(n) * (ols_estimator(zs) - rho_z)
    plt.hist(xs, bins=20, alpha=0.3, label='x')
    plt.hist(ys, bins=20, alpha=0.3, label='y')
    plt.hist(zs, bins=20, alpha=0.3, label='z')
    plt.legend(loc='upper left')
    plt.show()


if __name__=='__main__':
    #main()
    #print(generate_ar1_data(20, 0.7))
    check_distribution()