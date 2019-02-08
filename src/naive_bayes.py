from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class BernoulliNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, smooth=1):
        self.smooth = smooth

    def fit(self, X, y):
        n, m = X.shape
        assert y.shape == (n, 1), "Incorrect target dimension"

        # clazz = np.unique(y)  # target input might not be only binary
        clazz = np.array([0, 1])

        self.conditional = np.zeros((len(clazz), m))
        self.prior = np.zeros(len(clazz))

        for i, c in enumerate(clazz):
            nc = len(y[y == c])
            self.prior[i] = nc / n
            for j in range(m):
                class_only = X[np.argwhere(y.reshape(n) == c), j]
                self.conditional[i, j] = (len(class_only[class_only == 1]) + self.smooth) / (
                        nc + (m * self.smooth))

    def predict(self, X):
        wj1 = np.log(self.conditional[1] / self.conditional[0])
        wj0 = np.log((1 - self.conditional[1]) / (1 - self.conditional[0]))
        boundary = (wj0 * (1 - X) + wj1 * X).sum(axis=1) + np.log(self.prior[1]) - np.log(self.prior[0])
        return np.where(boundary > 0, 1, 0)[:, np.newaxis]
