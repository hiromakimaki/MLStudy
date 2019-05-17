import numpy as np
from scipy import stats

np.random.seed(0)


def one_sample_t_test():
    """
    Case 1:
        x ~ N(3, 1)
        Null hypothesis: `mu = 3`
    """
    n_iter = 1000
    n = 50
    alpha = 0.05
    print('Alpha: {}'.format(alpha))
    mu_null_hyp = 3
    t_values = np.zeros(n_iter)
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = np.random.normal(3, 1, n)
        t_value = np.sqrt(n) * (np.mean(xs) - mu_null_hyp) / np.std(xs)
        t_values[i] = t_value
        lower, upper = stats.t.ppf(q=[alpha/2, 1-alpha/2], df=n)
        if t_value < lower or upper < t_value:
            rejected[i] = 1
    print('Null hypothesis rejected: {} / {}'.format(rejected.sum(), len(rejected)))


def two_sample_t_test():
    """
    Case 1: same variances
        x ~ N(10, 1), y ~ N(10, 1)
            (mu_x = 10, mu_y = 10, v_x = 1, v_y = 1)
        Null hypothesis: `mu_x = mu_y`
    """
    n_iter = 1000
    nx = 50
    ny = 60
    alpha = 0.05
    print('Alpha: {}'.format(alpha))
    t_values = np.zeros(n_iter)
    rejected = np.zeros(n_iter)
    for i in range(n_iter):
        xs = np.random.normal(10, 1, nx)
        ys = np.random.normal(10, 1, ny)    
        t_value = (np.mean(xs) - np.mean(ys)) / np.sqrt((1/nx + 1/ny) * ((nx - 1) * (np.std(xs)**2) + (ny - 1) * (np.std(ys)**2)) / (nx + ny - 2))
        t_values[i] = t_value
        lower, upper = stats.t.ppf(q=[alpha/2, 1-alpha/2], df=n)
        if t_value < lower or upper < t_value:
            rejected[i] = 1
    print('Null hypothesis rejected: {} / {}'.format(rejected.sum(), len(rejected)))



def main():
    one_sample_t_test()
    two_sample_t_test()


if __name__=='__main__':
    main()