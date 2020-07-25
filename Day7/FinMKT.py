import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
# some usable model
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')


def data_preprocess(data):
    digital_cols = data.dtypes[data.dtypes != 'object'].index
    data[digital_cols] = data[digital_cols].apply(
        lambda x: (x - x.mean()) / (x.std()))
    x = pd.get_dummies(data)
    return x


def split_data(data):
    y = data.y
    x = data_preprocess(data.loc[:, data.columns != 'y'])
    # x = data_preprocess(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test


def print_result(name, y_test, y_pred):
    report = confusion_matrix(y_test, y_pred)
    precision = report[1][1] / (report[:, 1].sum())
    recall = report[1][1] / (report[1].sum())
    print(name + ': precision:' + str(precision)[:4] + ' recall:' + str(recall)[:4] + ' F1: ' + str(2 * precision * recall / (precision + recall))[:4])


def predict_model(model, x_train, x_test, y_train):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred


def predict_MyPerceptor(x_train, x_test, y_train):
    w = func_per(np.array(x_train[data.dtypes[data.dtypes != 'object'].index].T), list(
        y_train.map(dict(yes=1, no=0))))
    y_pred = x_test[data.dtypes[data.dtypes != 'object'].index].T * w
    return y_pred


def func_per(X, y, dim=100):
    P, N = X.shape
    w = np.zeros((P+1, 1))
    for iD in range(dim):
        for iN in range(N):
            y_try = w[0][0]
            for iP in range(0, P):
                y_try += X[iP][iN] * w[iP+1][0]
            if((y_try * y[iN]) <= 0):  # Unmatch
                w[0][0] += y[iN]
                for iP in range(P):    # Update all weights
                    w[iP+1][0] += X[iP][iN] * y[iN]
    return w[1:P+1]


if __name__ == '__main__':
    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    print_report = False

    y_pred = predict_model(LogisticRegression(), x_train, x_test, y_train)
    print_result('LogisticRegression', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(SVC(kernel='sigmoid', C=0.1),
                           x_train, x_test, y_train)
    print_result('SVM', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(Perceptron(), x_train, x_test, y_train)
    print_result('SKPerceptron', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(KNeighborsClassifier(
        n_neighbors=30, weights='distance'), x_train, x_test, y_train)
    print_result('KNeighborsClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(DecisionTreeClassifier(
        criterion='gini'), x_train, x_test, y_train)
    print_result('DecisionTreeClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(MLPClassifier(hidden_layer_sizes=(
        200, 200), solver='adam', shuffle=True), x_train, x_test, y_train)
    print_result('MLPClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(XGBClassifier(
        learning_rate=0.3, max_depth=3), x_train, x_test, y_train)
    print_result('XGBClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(RandomForestClassifier(
        n_estimators=200, criterion='gini'), x_train, x_test, y_train)
    print_result('RandomForestClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(AdaBoostClassifier(), x_train, x_test, y_train)
    print_result('AdaBoostClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(GradientBoostingClassifier(),
                           x_train, x_test, y_train)
    print_result('GradientBoostingClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    y_pred = predict_model(BaggingClassifier(), x_train, x_test, y_train)
    print_result('BaggingClassifier', y_test, y_pred)
    if(print_report):
        print(classification_report(y_test, y_pred))

    # y_pred = predict_MyPerceptor(x_train, x_test, y_train)
    # print_result('MyPerceptor', y_test, y_pred)
