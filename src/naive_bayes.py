from sklearn.base import BaseEstimator, ClassifierMixin


class BernoulliNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        print("")

    def fit(self, X, y):
        print(X.shape, y.shape)

    def predict(self, X):
        print(X.shape)


b = BernoulliNaiveBayes()
