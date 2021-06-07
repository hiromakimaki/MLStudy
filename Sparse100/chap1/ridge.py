import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)


class MyLinearRidge:

    def __init__(self, alpha=0, fit_intercept=True):
        self._alpha = alpha
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        n, p = X.shape
        assert n == y.shape[0]
        mat = np.dot(X.T, X) + self._alpha * n * np.eye(p)
        vec = np.dot(X.T, y)
        beta = np.linalg.solve(mat, vec)
        self.coef_ = beta.copy()
        if self._fit_intercept:
            self.intercept_ = y.mean() - np.dot(X, beta).mean()
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
    betas = np.zeros((alphas.shape[0], p + 1))  # with intercept
    for i, alpha in enumerate(alphas):
        print(f'{i}-th iteration started.')
        model = MyLinearRidge(alpha=alpha, fit_intercept=True)
        model.fit(X, y)
        betas[i, 0] = model.intercept_
        betas[i, 1:] = model.coef_.copy()
    for j in range(p + 1):
        plt.plot(alphas, betas[:, j], label=f'$\\beta_{j}$')
    plt.legend(loc='upper right')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.title('Ridge path')
    plt.show()


if __name__ == '__main__':
    main()
