import numpy
import pandas
from math import e

class Node:
    def __init__(self, x, gradient, hessian, idxs, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=10, lambda_=1, gamma=1, eps=0.1):

        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = numpy.random.permutation(
            self.col_count)[:round(self.subsample_cols*self.col_count)]

        self.val = self.compute_gamma(
            self.gradient[self.idxs], self.hessian[self.idxs])

        self.score = float('-inf')
        self.find_varsplit()

    def compute_gamma(self, gradient, hessian):
        '''
        Calculates the optimal leaf value equation (5)
        '''
        return (-numpy.sum(gradient)/(numpy.sum(hessian) + self.lambda_))

    def find_varsplit(self):
        '''
        Scans through every column and calcuates the best split point.
        The node is then split at this point and two new nodes are created.
        Depth is only parameter to change as we have added a new layer to tre structure.
        If no split is better than the score initalised at the begining then no splits further splits are made
        '''
        for c in self.column_subsample:
            self.find_greedy_split(c)
        if self.is_leaf:
            return
        x = self.split_col
        lhs = numpy.nonzero(x <= self.split)[0]
        rhs = numpy.nonzero(x > self.split)[0]
        self.lhs = Node(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[lhs], min_leaf=self.min_leaf, depth=self.depth-1,
                        lambda_=self.lambda_, gamma=self.gamma, min_child_weight=self.min_child_weight, eps=self.eps, subsample_cols=self.subsample_cols)
        self.rhs = Node(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[rhs], min_leaf=self.min_leaf, depth=self.depth-1,
                        lambda_=self.lambda_, gamma=self.gamma, min_child_weight=self.min_child_weight, eps=self.eps, subsample_cols=self.subsample_cols)

    def find_greedy_split(self, var_idx):
        '''
         For a given feature greedily calculates the gain at each split.
         Globally updates the best score and split point if a better split point is found
        '''
        x = self.x[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]

            lhs_indices = numpy.nonzero(x <= x[r])[0]
            rhs_indices = numpy.nonzero(x > x[r])[0]
            if (rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf
               or self.hessian[lhs_indices].sum() < self.min_child_weight
               or self.hessian[rhs_indices].sum() < self.min_child_weight):
                continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def gain(self, lhs, rhs):
        '''
        Calculates the gain at a particular split point bases on equation (7)
        '''
        gradient = self.gradient[self.idxs]
        hessian = self.hessian[self.idxs]

        lhs_gradient = gradient[lhs].sum()
        lhs_hessian = hessian[lhs].sum()

        rhs_gradient = gradient[rhs].sum()
        rhs_hessian = hessian[rhs].sum()

        gain = 0.5 *((lhs_gradient**2/(lhs_hessian + self.lambda_)) + (rhs_gradient**2/(rhs_hessian + self.lambda_)) - ((lhs_gradient + rhs_gradient)**2/(lhs_hessian + rhs_hessian + self.lambda_))) - self.gamma
        return (gain)

    @property
    def split_col(self):
        '''
        splits a column
        '''
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        '''
        checks if node is a leaf
        '''
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return numpy.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return (self.val)

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)
