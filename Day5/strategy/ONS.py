# ONS Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
import cvxpy


span_t = 120


def proj_netwon(A, y):
    n = A.shape[0]
    x = cvxpy.Variable(n)
    objective = cvxpy.Minimize(cvxpy.quad_form(x - y, A))
    constraints = [x >= 0, cvxpy.sum(x) == 1]
    prob = cvxpy.Problem(objective, constraints)
    prob.solve()
    return x.value


def ONS_weight_compute(n, context, w_pre, Ak_inv, eta=0.02, epsilon=0.125):
    x_k = np.array(context["Rk"])
    w = w_pre
    Ak = epsilon * np.eye(n)
    grad_k = - x_k / np.dot(w, x_k)
    hess_k = np.outer(grad_k, grad_k)
    Ak += hess_k
    Ak_inv -= Ak_inv.dot(hess_k).dot(Ak_inv) / \
        (1 + grad_k.dot(Ak_inv).dot(grad_k))
    w = proj_netwon(Ak, w - eta * Ak_inv.dot(grad_k))

    return w, Ak_inv


if __name__ == "__main__":
    print("this is ONS Portfolio")
