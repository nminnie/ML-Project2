import numpy as np
import unittest
from src.naive_bayes import BernoulliNaiveBayes


class TestBNN(unittest.TestCase):

    def testBool(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([[0], [0], [1], [1]])

        X_new = np.array([[1, 0], [1, 1]])
        y_exact = np.array([[0], [1]])

        nv = BernoulliNaiveBayes()
        nv.fit(X, y)
        y_pred = nv.predict(X_new)

        self.assertTrue((y_pred == y_exact).all())
