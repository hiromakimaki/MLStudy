import numpy as np


np.random.seed(0)


class MyLinearLasso:

    def __init__(self, alpha=0, fit_intercept=True):
        self._alpha = alpha
        self._fit_intercept = fit_intercept

    def soft_threshold(self, x):
        return np.sign(x) * np.clip(np.abs(x) - self._alpha, 0, None)

    def fit(self, X, y, max_iter=1000):
        if self._fit_intercept:
            x = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            x = X.copy()
        n, p = x.shape
        assert n == y.shape[0]
        beta, beta_old = np.zeros(p), np.zeros(p)
        iter_count = 0
        while True:
            for j in range(p):
                r = y - (np.dot(x, beta) - x[:, j] * beta[j])
                s = np.dot(x[:, j], r) / n
                beta[j] = self.soft_threshold(s) / (np.sum(x[:, j]**2) / n)
            eps = np.linalg.norm(beta - beta_old)
            if eps < 10**(-6):
                break
            iter_count += 1
            if iter_count >= max_iter:
                print('iteration stopped because iteration count attains `max_iter`')
            beta_old = beta.copy()
        return beta


def generate_sample_data():
    n, p = 100, 5
    X = np.random.normal(0, 1, n * p).reshape((n, p))
    beta = np.arange(p)
    y = np.dot(X, beta)
    return X, y


def main():
    model = MyLinearLasso(alpha=1.0, fit_intercept=False)
    X, y = generate_sample_data()
    beta = model.fit(X, y)
    print(beta)


if __name__ == '__main__':
    main()
