import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func_SVM, func_per, func_lin

fp = open("Test_Result.txt", "w")
fp.close()
fp = open("Test_Result.txt", "a")


def main_func(no_train=500, no_test=100):
    no_iter = 500

    no_data = no_train + no_test  # number of all data
    assert(no_train + no_test == no_data)

    cumulative_train_err = 0
    cumulative_test_err = 0

    for i in range(no_iter):
        X, y, w_f = gen_data(no_data)

        X_train, X_test = X[:, :no_train], X[:, no_train:]
        y_train, y_test = y[:, :no_train], y[:, no_train:]
        w_g = func_lin(X_train, y_train)
        # w_g = func_SVM(X_train, y_train)
        # w_g = func_per(X_train, y_train)


        # Compute training, testing error
        # YOUR CODE HERE
        # ----------------
        # ANSWER BEGIN
        # ----------------

        train_err = 0
        test_err = 0
        for j in range(no_train):
            if((X_train[0][j] * w_g[1][0] + X_train[1][j] * w_g[2][0] + w_g[0][0]) * y_train[0][j] <= 0):
                train_err = 1
                break

        for j in range(no_test):
            if((X_test[0][j] * w_g[1][0] + X_test[1][j] * w_g[2][0] + w_g[0][0]) * y_test[0][j] <= 0):
                test_err = 1
                break

        # ----------------
        # ANSWER END
        # ----------------
        cumulative_train_err += train_err
        cumulative_test_err += test_err

    train_err = cumulative_train_err / no_iter
    test_err = cumulative_test_err / no_iter

    fp.writelines("Iter: {:d}, No_train: {:d}, No_test: {:d}\n".format(
        no_iter, no_train, no_test))
    fp.writelines("Training error: %s\n" % train_err)
    fp.writelines("Testing error: %s\n\n"% test_err)
    plot(X, y, w_f, w_g, "Linear"+'_'+str(no_train)+'_'+str(no_test))
    # plot(X, y, w_f, w_g, "SVM"+'_'+str(no_train)+'_'+str(no_test))
    # plot(X, y, w_f, w_g, "Perceptron"+'_'+str(no_train)+'_'+str(no_test))
    # plot(X, y, w_f, w_g, "Perceptron")


if __name__ == "__main__":
    # main_func(100, 900)
    # main_func(200, 800)
    # main_func(300, 700)
    # main_func(400, 600)
    # main_func(500, 500)
    # main_func(600, 400)
    # main_func(700, 300)
    main_func(800, 200)
    # main_func(900, 100)
