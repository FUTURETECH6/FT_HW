import numpy as np
import matplotlib.pyplot as plt


def plot(X, y, wf, wg, title):
    '''
    Plot data set.

    INPUT:  X: sample features, P-by-N matrix.
            y: sample labels, 1-by-N row vector.
            wf: true target function parameters, (P+1)-by-1 column vector.
            wg: learnt target function parameters, (P+1)-by-1 column vector.
            title: title of figure.
    '''

    if X.shape[0] != 2:
        print('Here we only support 2-d X data')
        return

    plt.plot(X[0, y.flatten() == 1], X[1, y.flatten() == 1], 'o', markerfacecolor='r',
             markersize=10)

    plt.plot(X[0, y.flatten() == -1], X[1, y.flatten() == -1], 'o', markerfacecolor='g',
             markersize=10)

    k, b = -wf[1] / wf[2], -wf[0] / wf[2]
    max_x = max(min((1 - b) / k, (-1 - b) / k), -1)
    min_x = min(max((1 - b) / k, (-1 - b) / k), 1)
    x = np.arange(min_x, max_x, (max_x - min_x) / 100)
    temp_y = k * x + b
    plt.plot(x, temp_y, color='b', linewidth=2, linestyle='-')
    k, b = -wg[1] / wg[2], -wg[0] / wg[2]
    max_x = max(min((1 - b) / k, (-1 - b) / k), -1)
    min_x = min(max((1 - b) / k, (-1 - b) / k), 1)
    x = np.arange(min_x, max_x, (max_x - min_x) / 100)
    temp_y = k * x + b
    plt.plot(x, temp_y, color='b', linewidth=2, linestyle='--')
    plt.title(title)
    # plt.show()
    plt.savefig('./Results_Plots/'+title+".svg", format="svg")
    plt.close()
