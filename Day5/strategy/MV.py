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
    covMat = np.cov(context["R"].T)
    meanMat = np.mean(context["R"].T, axis = 1)

    P = 2 * matrix(covMat)
    q = -1 * matrix(meanMat)

    G = -1 * matrix(np.eye(n))
    h = matrix(0.0, (n, 1))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x']).T[0]

    # w = np.zeros(n)
    # R = context["R"].T
    
    # N = 100
    # mus = [10**(5.0*t/N-1.0) for t in range(N)]

    # S = matrix(np.cov(R))
    # pbar = matrix(np.mean(R,axis=1))

    # G = -matrix(np.eye(n))
    # h = matrix(0.0, (n,1))
    # A = matrix(1.0, (1,n))
    # b = matrix(1.0)

    # solvers.options['show_progress'] = False
    # prtflio = [solvers.qp(mu*S,-pbar, G, h ,A, b)['x'] for mu in mus]#有效边界
    # re = [blas.dot(pbar,x) for x in prtflio]#有效边界的收益
    # risk = [np.sqrt(blas.dot(x, S*x)) for x in prtflio]#有效边界的收益
    # m1 = np.polyfit(re, risk, 2)
    # x1 = np.sqrt(m1[2]/m1[0])
    # w = np.asarray(np.squeeze(solvers.qp(matrix(x1*S), -pbar, G, h, A, b)['x']))

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