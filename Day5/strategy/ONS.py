# ONS Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def ONS_weight_compute(n, context, w_pre=None, Eta=0.4, epsilon=0.01):
    diff = np.array(context["Rk"] - 1)
    A_i_inv = (1 / epsilon) * np.eye(m)
    A_i = epsilon * np.eye(m)
    # update w_i
    grad_i = grad(w_pre, diff)
    hess_i = np.outer(grad_i, grad_i)
    A_i += hess_i
    A_i_inv -= A_i_inv.dot(hess_i).dot(A_i_inv) / (1 + grad_i.dot(A_i_inv).dot(grad_i))
    w_k = proj_netwon(A_i, w_pre - Eta * A_i_inv.dot(grad_i))

    return w_k


if __name__ == "__main__":
    print("this is ONS Portfolio")

