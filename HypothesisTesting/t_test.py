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


def main():
    one_sample_t_test()


if __name__=='__main__':
    main()