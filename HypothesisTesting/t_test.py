import numpy as np
from scipy import stats

np.random.seed(0)


def one_sample_t_test():
    """
    Case 1:
        x ~ N(3, 1)
        Null hypothesis: `mu = 3`
    """
    n = 50
    xs = np.random.normal(3, 1, n)

    mu_null_hyp = 3
    t_value = np.sqrt(n) * (np.mean(xs) - mu_null_hyp) / np.std(xs)
    print(np.mean(xs))
    print('T-value: {}'.format(t_value))
    
    alpha = 0.05 / 2
    lower, upper = stats.t.ppf(q=[alpha, 1-alpha], df=n)
    print(lower, upper)
    if lower <= t_value <= upper:
        print('Not rejected.')
    else:
        print('Rejected.')


def two_sample_t_test():
    """
    Case 1: same variances
        x ~ N(10, 1), y ~ N(10, 1)
            (mu_x = 10, mu_y = 10, v_x = 1, v_y = 1)
        Null hypothesis: `mu_x = mu_y`
    """
    nx = 50
    ny = 60
    xs = np.random.normal(10, 1, nx)
    ys = np.random.normal(10, 1, ny)    
    t_value = (np.mean(xs) - np.mean(ys)) / np.sqrt((1/nx + 1/ny) * ((nx - 1) * (np.std(xs)**2) + (ny - 1) * (np.std(ys)**2)) / (nx + ny - 2))
    print('(Mean) Class 1: {}, Class 2: {}'.format(np.mean(xs), np.mean(ys)))
    print('(Std.) Class 1: {}, Class 2: {}'.format(np.std(xs), np.std(ys)))
    print('T-value: {}'.format(t_value))
    alpha = 0.05 / 2
    lower, upper = stats.t.ppf(q=[alpha, 1-alpha], df=(nx + ny - 2))
    print(lower, upper)
    if lower <= t_value <= upper:
        print('Not rejected.')
    else:
        print('Rejected.')


def main():
    one_sample_t_test()
    two_sample_t_test()


if __name__=='__main__':
    main()