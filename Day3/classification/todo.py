import numpy as np
from scipy.optimize import minimize

# You may find below useful for Support Vector Machine
# More details in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#from scipy.optimize import minimize

def func_lin(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned perceptron parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))

    # YOUR CODE HERE
    # ----------------
    # ANSWER BEGIN
    # ----------------

    x = np.vstack((np.ones((1, X.shape[1])), X))
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x, x.T)), x), y.T)

    # ----------------
    # ANSWER END
    # ----------------
    return w


def func_per(X, y, dim = 20):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned perceptron parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))

    # YOUR CODE HERE
    # ----------------
    # ANSWER BEGIN
    # ----------------

    for iD in range(dim):
        for iN in range(N):
            y_try = w[0][0]
            for iP in range(P):
                y_try += X[iP][iN] * w[iP+1][0]
            if((y_try * y[0][iN]) <= 0):  # Unmatch
                w[0][0] += y[0][iN]
                for j in range(P):        # Update all weights
                    w[j+1][0] += X[j][iN] * y[0][iN]

    # ----------------
    # ANSWER END
    # ----------------
    return w

def func_SVM(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned perceptron parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))

    # YOUR CODE HERE
    # ----------------
    # ANSWER BEGIN
    # ----------------

    x_i = np.vstack((np.ones((1, N)), X))
    con = {'type': 'ineq', 'fun': lambda w, X, y: np.multiply(y[0, :], np.matmul(w.T, X)) - 1, 'args': (x_i, y)}
    res = minimize(fun=lambda w : 0.5 * np.linalg.norm(w[1:,]) * np.linalg.norm(w[1:,]), x0=w, constraints=con, method='SLSQP')
    w = res.x.reshape(3, 1)

    # ----------------
    # ANSWER END
    # ----------------
    return w
