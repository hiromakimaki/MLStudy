import numpy as np


np.random.seed(0)


class MyLogisticRegressor:

    def __init__(self, max_iter=1000, fit_intercept=True):
        self._max_iter = max_iter
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        if self._fit_intercept:
            new_x = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            new_x = X.copy()
        beta = np.zeros(new_x.shape[1])
        for _ in range(self._max_iter):
            v = np.exp(- y * (np.dot(new_x, beta)))
            u = y * v / (1 + v)
            W = np.diag(v / (1 + v)**2)
            delta = np.linalg.solve(new_x.T @ W @ new_x, new_x.T @ u)
            beta += delta
            if np.linalg.norm(delta) > 10**(-5):
                break
        if self._fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:].copy()
        else:
            self.coef_ = beta.copy()
        return


# From chapter 1 script
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
        if self._fit_intercept:
            self.intercept_ = y.mean() - np.dot(X, beta).mean()
        return


class MyLogisticLasso:

    def __init__(self, alpha=0, max_iter=1000, fit_intercept=True):
        self._alpha = alpha
        self._max_iter = max_iter
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        # 近接newton法
        p = X.shape[1]
        iter_count = 0
        beta, new_beta = np.zeros(p), np.zeros(p)
        intercept, new_intercept = 0, 0
        lasso = MyLinearLasso(alpha=self._alpha, fit_intercept=False)
        while True:
            # betaに基づきW, zの更新
            v = np.exp(- y * (intercept + np.dot(X, beta)))
            u = y * v / (1 + v)
            W = np.diag(v / (1 + v)**2)
            z = intercept + np.dot(X, beta) + np.dot(np.linalg.inv(W), u)
            # W, zでbetaの更新
            XX = np.dot(np.sqrt(W), X)
            yy = np.dot(np.sqrt(W), z)
            lasso.fit(XX, yy)
            new_beta = lasso.coef_.copy()
            if self._fit_intercept:
                new_intercept = z.mean() - np.dot(X, new_beta).mean()
            if np.sqrt(np.linalg.norm(new_beta - beta)**2 + (new_intercept - intercept)**2) > 10**(-5):
                break
            beta = new_beta.copy()
            intercept = new_intercept
            iter_count += 1
            if iter_count >= self._max_iter:
                print('iteration stopped because iteration count attained `max_iter`')
                break
        if self._fit_intercept:
            self.intercept_ = new_intercept
        self.coef_ = new_beta.copy()
        return


def generate_sample_data(n=500, p=4):
    X = np.random.randn(n, p)
    beta = np.random.randn(p)
    s = np.dot(X, beta)
    prob = 1 / (1 + np.exp(-s))
    y = np.ones(n)
    y[np.random.rand(n) > prob] = -1
    return X, y, beta


def main():
    X, y, beta = generate_sample_data()
    print(beta)
    model = MyLogisticRegressor()
    model.fit(X, y)
    print(model.intercept_, model.coef_)
    new_model = MyLogisticLasso(alpha=0.05)
    new_model.fit(X, y)
    print(new_model.intercept_, new_model.coef_)


if __name__ == '__main__':
    main()
