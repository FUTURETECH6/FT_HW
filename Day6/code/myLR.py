'''
risk model
'''

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


class myLogisticRegression:
    def __init__(self):
        self.w = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_gradient(x):
        return myLogisticRegression.sigmoid(x) * (1 - myLogisticRegression.sigmoid(x))

    @staticmethod
    def ones_augment_to_left(X):
        X = np.array(X)
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])

    @staticmethod
    def logistic_gradient_descent(X, y, n_iters=10000, alpha=0.05, weight=None):
        w = weight
        if w is None:
            w = np.random.rand(X.shape[1])
            w = np.ones(X.shape[1])
        pass

        for i in range(1, n_iters + 1):
            y_pred = myLogisticRegression.sigmoid(X.dot(w))
            loss = y_pred - y

            grad = myLogisticRegression.sigmoid_gradient(
                loss.dot(X) / X.shape[0])
            w = w - alpha * grad
        return w

    def fit(self, X_train, y_train, **kwargs):
        X = self.ones_augment_to_left(X_train)
        y = np.array(y_train)
        self.w = self.logistic_gradient_descent(X, y, **kwargs)

        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        augX_test = self.ones_augment_to_left(X_test)
        return augX_test.dot(self.w)

    def score(self, X, Y):
        y_pred = self.predict(X)
        return(roc_auc_score(Y, np.round(y_pred)))


exclude_attr = []

file_list = [('Unprocessed', './data/train_raw.csv', './data/test_raw.csv'), ('Normalization', './data/train_nor.csv', './data/test_nor.csv'),
             ('WOE', './data/train_woe.csv', './data/test_woe.csv'), ('CrossFeatures', './data/train_cross.csv', './data/test_cross.csv')]

if __name__ == "__main__":
    for f_name, train_file, test_file in file_list:
        raw_train = pd.read_csv(train_file).astype('float')
        raw_test = pd.read_csv(test_file).astype('float')

        train_X = raw_train.drop(
            columns=['Y'] + exclude_attr)
        test_X = raw_test.drop(
            columns=['Y'] + exclude_attr)

        myLR = myLogisticRegression()
        myLR.fit(train_X, raw_train['Y'])
        print(f_name, 'eval auc:', str(myLR.score(test_X, raw_test['Y']))[:6])

        # raw_pre = pd.read_csv('../Day2/data/test_new.csv').astype('float')
        # pre = raw_pre.drop(columns=['Unnamed: 0', 'id'] + exclude_attr)
        # for i in pre.columns:
        #     pre[i].fillna(pre[i].mean(), inplace=True)
        # pd.DataFrame({'id': raw_test['id'], 'pre': LR.predict_proba(pre)}).to_csv("./3180103012_pre_LR.csv")
