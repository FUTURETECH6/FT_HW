'''
risk model
'''

import lightgbm as lgb
import pandas as pd
import sklearn

exclude_attr = []

file_list = [('Unprocessed', 'raw'), ('Normalization', 'nor'), ('WOE', 'woe'), ('CrossFeatures', 'cross')]

class RiskModel():
    def __init__(self, data_path, ftype):
        self.data_path = data_path
        self.train, self.test, self.param = self.__construct_dataset(ftype)

        # self.train = self.train.drop(columns=['Unnamed: 0', 'id'] + exclude_attr)
        # self.test = self.test.drop(columns=['Unnamed: 0', 'id'] + exclude_attr)

        self.feature_name = [i for i in self.train.columns if i not in ['Y']]
        # print('train set:', self.train.shape, ', ', 'test set:', self.test.shape)
        self.lgb_train = lgb.Dataset(data=self.train[self.feature_name],
                                     label=self.train['Y'],
                                     feature_name=self.feature_name,
                                     silent=True)
        self.lgb_test = lgb.Dataset(data=self.test[self.feature_name],
                                    label=self.test['Y'],
                                    feature_name=self.feature_name,
                                    silent=True)
        self.evals_result = {}
        self.gbm = None

    def __construct_dataset(self, ftype):
        train = pd.read_csv(self.data_path + 'train_' + ftype + '.csv')
        test = pd.read_csv(self.data_path + 'test_' + ftype + '.csv')

        train = train.astype('float')
        test = test.astype('float')

        param = dict()
        param['verbose'] = -1
        param['objective'] = 'binary'
        param['boosting_type'] = 'gbdt'
        param['metric'] = 'auc'
        param['verbose'] = 0
        param['learning_rate'] = 0.10
        param['max_depth'] = -1
        param['feature_fraction'] = 0.8
        param['bagging_fraction'] = 0.8
        param['bagging_freq'] = 1
        param['num_leaves'] = 15
        param['min_data_in_leaf'] = 64
        param['is_unbalance'] = False
        param['verbose'] = -1

        return train, test, param

    def fit(self):
        self.gbm = lgb.train(self.param,
                             self.lgb_train,
                             early_stopping_rounds=10,
                             num_boost_round=1000,
                             evals_result=self.evals_result,
                             valid_sets=[self.lgb_train, self.lgb_test],
                             verbose_eval=False)

    def evaluate(self):
        test_label = self.test['Y']
        prob_label = self.gbm.predict(self.test[self.feature_name])
        auc = sklearn.metrics.roc_auc_score(test_label, prob_label)
        return auc

if __name__ == "__main__":
    for f_name, ftype in file_list:
        MODEL = RiskModel(data_path='./data/', ftype=ftype)
        MODEL.fit()
        print(f_name, 'eval auc:', str(MODEL.evaluate())[:6])

        # raw_pre = pd.read_csv('../Day2/data/test_new.csv')
        # pre = raw_pre.drop(columns=['id'] + exclude_attr)
        # pd.DataFrame({'id': raw_pre['id'], 'pre': MODEL.gbm.predict(pre)}).to_csv("./3180103012_pre_LGB.csv")