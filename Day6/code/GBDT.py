'''
risk model
'''

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


exclude_attr = []

if __name__ == "__main__":
    raw_train = pd.read_csv('./data/train.csv').astype('float')
    raw_test = pd.read_csv('./data/test.csv').astype('float')

    train_X = raw_train.drop(columns=['Unnamed: 0', 'id', 'Y'] + exclude_attr)
    test_X = raw_test.drop(columns=['Unnamed: 0', 'id', 'Y'] + exclude_attr)

    LR = GradientBoostingClassifier(loss='deviance', learning_rate=0.3)
    LR.fit(train_X, raw_train['Y'])
    print('eval auc:', LR.score(test_X, raw_test['Y']))

    # raw_pre = pd.read_csv('../Day2/data/test_new.csv').astype('float')
    # pre = raw_pre.drop(columns=['Unnamed: 0', 'id'] + exclude_attr)
    # for i in pre.columns:
    #     pre[i].fillna(pre[i].mean(), inplace=True)
    # pd.DataFrame({'id': raw_test['id'], 'pre': LR.predict_proba(pre)}).to_csv("./3180103012_pre_GBDT.csv")