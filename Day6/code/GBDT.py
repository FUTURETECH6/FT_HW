'''
risk model
'''

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

exclude_attr = []

file_list = [('Normalization', './data/train_nor.csv', './data/test_nor.csv'), ('WOE', './data/train_woe.csv', './data/test_woe.csv'),
             ('Cross Features', './data/train_cross.csv', './data/test_cross.csv')]

if __name__ == "__main__":
    for f_name, train_file, test_file in file_list:
        raw_train = pd.read_csv(train_file).astype('float')
        raw_test = pd.read_csv(test_file).astype('float')

        train_X = raw_train.drop(
            columns=['Unnamed: 0', 'Y'] + exclude_attr)
        test_X = raw_test.drop(
            columns=['Unnamed: 0', 'Y'] + exclude_attr)

        LR = GradientBoostingClassifier(loss='deviance', learning_rate=0.3)
        LR.fit(train_X, raw_train['Y'])
        print(f_name, 'eval auc:', LR.score(test_X, raw_test['Y']))

        # raw_pre = pd.read_csv('../Day2/data/test_new.csv').astype('float')
        # pre = raw_pre.drop(columns=['Unnamed: 0', 'id'] + exclude_attr)
        # for i in pre.columns:
        #     pre[i].fillna(pre[i].mean(), inplace=True)
        # pd.DataFrame({'id': raw_test['id'], 'pre': LR.predict_proba(pre)}).to_csv("./3180103012_pre_GBDT.csv")
