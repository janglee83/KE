from Node import Node
import numpy

class XGBoostTree:
    def fit(self, x, gradient, hessian, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=10, lambda_=1, gamma=1, eps=0.1):
        self.dtree = Node(x, gradient, hessian, numpy.array(numpy.arange(
            len(x))), subsample_cols, min_leaf, min_child_weight, depth, lambda_, gamma, eps)
        return self

    def predict(self, X):
        return self.dtree.predict(X)
