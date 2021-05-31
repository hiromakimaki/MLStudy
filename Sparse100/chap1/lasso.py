import numpy as np
import matplotlib.pyplot as plt


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
                print('iteration stopped because iteration count attained `max_iter`')
            beta_old = beta.copy()
        return beta


def generate_sample_data():
    n, p = 100, 5
    X = np.random.normal(0, 1, n * p).reshape((n, p))
    beta_sign = (-1) ** np.arange(p)
    beta = beta_sign * np.arange(1, p + 1)
    y = np.dot(X, beta)
    return X, y


def main():
    X, y = generate_sample_data()
    n, p = X.shape
    alphas = np.arange(0.05, 5, 0.05)
    betas = np.zeros((alphas.shape[0], p))
    for i, alpha in enumerate(alphas):
        print(f'{i}-th iteration started.')
        model = MyLinearLasso(alpha=alpha, fit_intercept=False)
        beta = model.fit(X, y)
        betas[i, :] = beta.copy()
    for j in range(p):
        plt.plot(alphas, betas[:, j], label=f'$\\beta_{j}$')
    plt.legend(loc='upper right')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.title('LASSO path')
    plt.show()


if __name__ == '__main__':
    main()
