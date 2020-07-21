import sklearn
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np


def classification(processed_df):
    # split into train and test sets
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    split_result = next(kf.split(processed_df), None)
    train = processed_df.iloc[split_result[0]]
    test = processed_df.iloc[split_result[1]]

    # TF-IDF
    X_train_tf, X_test_tf = TF_IDF(train, test)

    # classification
    # Your code here
    # Answer begin

    clf = MultinomialNB()
    clf.fit(X_train_tf, train['industry'])
    y_predict = clf.predict(X_test_tf)

    results = metrics.classification_report(test['industry'], y_predict)

    # Answer end
    return results


def TF_IDF(train, test):
    # Your code here
    # Answer begin

    vectorizer = CountVectorizer(min_df=0.01)
    transformer = TfidfTransformer()
    X_train_tf = transformer.fit_transform(
        vectorizer.fit_transform(train['business_scope'].values))
    X_test_tf = transformer.transform(
        vectorizer.transform(test['business_scope'].values))

    # Answer end
    return X_train_tf, X_test_tf
