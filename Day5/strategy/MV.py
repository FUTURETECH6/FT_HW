# MV Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
from cvxopt import solvers,matrix


span_t = 120


def MV_weight_compute(n, context):
    meanMat = np.mean(context["R"].T, axis = 1)
    covMat = np.cov(context["R"].T)

    P = 2 * matrix(covMat)
    Q = -1 * matrix(meanMat)

    G = -matrix(np.eye(n))
    h = matrix(0.0, (n, 1))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, Q, G, h, A, b)
    w = np.array(sol['x']).T
    w = w[0]

    return w


if __name__ == "__main__":
    print("this is MV Portfolio")
