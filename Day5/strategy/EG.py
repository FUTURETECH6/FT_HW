# EG Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def EG_weight_compute(n, context, w_pre=None, eta=0.05):
    x_k = np.array(context["Rk"])
    grad_k = - x_k / np.dot(w_pre, x_k)
    
    w_k = w_pre * np.exp(-eta * grad_k)
    w_k = w_k / np.sum(w_k)
    # if np.isnan(w_k).any():
    #     w_k = np.zeros(n)
    #     w_k[np.argmax(context["Rk"])] = 1
    return w_k

if __name__ == "__main__":
    print("this is EG Portfolio")
