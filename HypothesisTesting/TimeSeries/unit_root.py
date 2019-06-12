import numpy as np
from scipy import stats


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


def check_distribution(n_iter=100):
    tau_x = np.zeros(n_iter)
    tau_y = np.zeros(n_iter)
    n = 20
    for i in range(n_iter):
        xs = generate_ar1_data(n, 0.7) # not unit root process
        tau_x[i] = np.sqrt(n) * (ols_estimator(xs) - 0.7)
        ys = generate_ar1_data(n, 1) # unit root process
        tau_y[i] = np.sqrt(n) * (ols_estimator(ys) - 1)
    print(tau_x)
    print(tau_y)


if __name__=='__main__':
    #main()
    #print(generate_ar1_data(20, 0.7))
    check_distribution()