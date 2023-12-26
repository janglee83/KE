import numpy
from math import e
from XGBoostTree import XGBoostTree

class XGBoostClassifier:
    def __init__(self):
        self.estimators = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + numpy.exp(-x))

    # first order gradient logLoss
    def grad(self, preds, labels):
        preds = self.sigmoid(preds)
        return (preds - labels)

    # second order gradient logLoss
    def hess(self, preds, labels):
        preds = self.sigmoid(preds)
        return (preds * (1 - preds))

    def fit(self, X, y, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5, learning_rate=0.4, boosting_rounds=5, lambda_=1.5, gamma=1, eps=0.1):
        self.X, self.y = X, y
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma

        self.base_pred = numpy.full(
            (X.shape[0], 1), 1).flatten().astype('float64')

        for _ in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth=self.depth, min_leaf=self.min_leaf, lambda_=self.lambda_,
                                              gamma=self.gamma, eps=self.eps, min_child_weight=self.min_child_weight, subsample_cols=self.subsample_cols)
            self.base_pred += self.learning_rate * \
                boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)

    def predict(self, X):
        pred = numpy.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)

        predicted_probas = self.sigmoid(
            numpy.full((X.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = numpy.where(predicted_probas > numpy.mean(predicted_probas), 1, 0)
        return (preds)
