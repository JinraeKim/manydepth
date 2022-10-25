from copy import copy

import numpy as np

from numpy.random import default_rng


class RANSAC:
    """
    Borrowed from https://en.wikipedia.org/wiki/Random_sample_consensus#Example_Code.
    The computation time will be proportional to `k`.
    """
    def __init__(self, n=10, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.inlier_indices = None
        self.rng = default_rng()

    def fit(self, X, y):
        if len(X) == 0:
            pass
        else:
            for _ in range(self.k):
                ids = self.rng.permutation(X.shape[0])
                n = self.n

                maybe_inliers = ids[:n]
                maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

                thresholded = (
                    self.loss(y[ids][n:], maybe_model.predict(X[ids][n:]))
                    < self.t
                )
                inlier_ids = ids[n:][np.flatnonzero(thresholded).flatten()]

                if inlier_ids.size > self.d:
                    inlier_ids_including_maybe = np.hstack([maybe_inliers, inlier_ids])
                    better_model = copy(self.model).fit(X[inlier_ids_including_maybe], y[inlier_ids_including_maybe])

                    this_error = self.metric(
                        y[inlier_ids_including_maybe], better_model.predict(X[inlier_ids_including_maybe])
                    )

                    if this_error < self.best_error:
                        self.best_error = this_error
                        self.best_fit = maybe_model
                        self.inlier_indices = inlier_ids
        return self

    def predict(self, X):
        return self.best_fit.predict(X)


def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2  # speicialised to deal with angles


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


class LinearRegressor:
    def __init__(self):
        self.params = None
        self.params_dim = (1+1, 1)  # in case of receiving no data; +1 is for bias
        self.success = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        try:
            self.params = np.linalg.inv(X.T @ X) @ X.T @ y
            self.success = True
        except np.linalg.LinAlgError as e:
            self.params = np.random.normal(size=np.prod(self.params_dim)).reshape(self.params_dim)
            print("maybe too small number of lines are provided, random parameter is generated;", e)
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params
