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
        n, p = X.shape
        assert n == y.shape[0]
        beta, beta_old = np.zeros(p), np.zeros(p)
        iter_count = 0
        while True:
            for j in range(p):
                r = y - (np.dot(X, beta) - X[:, j] * beta[j])
                s = np.dot(X[:, j], r) / n
                beta[j] = self.soft_threshold(s) / (np.sum(X[:, j]**2) / n)
            eps = np.linalg.norm(beta - beta_old)
            if eps < 10**(-6):
                break
            iter_count += 1
            if iter_count >= max_iter:
                print('iteration stopped because iteration count attained `max_iter`')
                break
            beta_old = beta.copy()
        self.coef_ = beta.copy()
        self.intercept_ = y.mean() - np.dot(X, beta).mean()
        return


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
    betas = np.zeros((alphas.shape[0], p + 1))  # with intercept
    for i, alpha in enumerate(alphas):
        print(f'{i}-th iteration started.')
        model = MyLinearLasso(alpha=alpha, fit_intercept=True)
        model.fit(X, y)
        betas[i, 0] = model.intercept_
        betas[i, 1:] = model.coef_.copy()
    for j in range(p + 1):
        plt.plot(alphas, betas[:, j], label=f'$\\beta_{j}$')
    plt.legend(loc='upper right')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.title('LASSO path')
    plt.show()


if __name__ == '__main__':
    main()
