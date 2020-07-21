# EG Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def exponentiated_gradient(data_set, previous_weight, learning_rate):

    result = []
    all_weighted_value = np.sum(
        [previous_weight[i] * data_set[i] for i in range(len(data_set))])
    numerator = np.sum([previous_weight[i] * np.exp((learning_rate *
                                                     data_set[i]) / all_weighted_value) for i in range(len(data_set))])

    for i in range(len(data_set)):
        fractions = previous_weight[i] * \
            np.exp((learning_rate * data_set[i]) / all_weighted_value)
        result.append(fractions / numerator)
    return result


def EG_weight_compute(n, context, learning_rate=0.3):
    DiffMat = np.array(context["R"].T - 1)

    w = np.array([1/n for i in range(n)], dtype=np.float64)
    for i in range(DiffMat.shape[1]):
        w = exponentiated_gradient(DiffMat[:, i], w, learning_rate)

    return w
    # w_i = np.ones(m) / m
    # f_values = []
    # for i in tqdm(range(n)):
    #     x_i = data[i, :]
    #     f_i = f(w_i, x_i)
    #     f_values.append(f_i)
    #     # update w_i
    #     grad_i = grad(w_i, x_i)
    #     w_i = w_i * np.exp(-eta * grad_i)
    #     w_i = w_i / np.sum(w_i)
    # return f_values


if __name__ == "__main__":
    print("this is EG Portfolio")
    NYSE = {"name": "NYSE", "filename": "NYSE.txt",
            "span_t": 120, "init_t": 20, "frequency": "none"}

    datasets = [NYSE]
    dataset_name = ["NYSE"]

    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mat_data_path = PARENT_DIR + '/data/ff25_input.mat'
    # mat_data_path = PARENT_DIR + '/data/ff48_input.mat'
    NYSE = Stocks(path=mat_data_path)
    m = NYSE.Nmonths
    n = NYSE.Nportfolios
    R = NYSE.portfolios
    portfolio = Portfolio(stock=NYSE)
    for k in range(span_t - 1, m, 1):
        wk = EG_weight_compute(n, context)
        portfolio.rebalance(target_weights=wk)
    print(portfolio.eval(portfolio.cumulative_wealth))
    print(portfolio.eval(portfolio.sharpe_ratio))
    print(portfolio.eval(portfolio.volatility))
    print(portfolio.eval(portfolio.max_drawdown))
    print(portfolio.eval(portfolio.turnover))
