# MV Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
from cvxopt import solvers, matrix, blas


span_t = 120


def MV_weight_compute(n, context):
    P = 2 * matrix(np.cov(context["R"].T))
    q = -1 * matrix(np.mean(context["R"].T, axis=1))
    G = -1 * matrix(np.eye(n))
    h = matrix(0.0, (n, 1))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x']).T[0]

    return w


'''
Input arguments.(n=36)

P is a n x n dense or sparse 'd' matrix with the lower triangular, part of P stored in the lower triangle. Must be positive semidefinite.

q is an n x 1 dense 'd' matrix.

G is an m x n dense or sparse 'd' matrix.

h is an m x 1 dense 'd' matrix.

A is a p x n dense or sparse 'd' matrix.

b is a p x 1 dense 'd' matrix or None.
'''


if __name__ == "__main__":
    print("this is MV Portfolio")
