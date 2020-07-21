import tushare as ts
import pandas as pd
import jieba
import jieba.analyse


def get_data(token, N):
    # get_data
    pro = ts.pro_api(token)
    pd.set_option('max_colwidth', 120)
    df0 = pro.stock_company(exchange='SZSE', fields='ts_code, business_scope')
    df1 = df0.dropna(axis=0, how='any')
    df2 = pro.stock_basic(exchange='SZSE', fields='ts_code, name, industry')

    # merge
    # Your code here
    # Answer begin

    pass

    # Answer end

    df = pd.merge(df1, df2, how='right')

    # filter by number of records
    nonan_df = df.dropna(axis=0, how='any')
    vc = nonan_df['industry'].value_counts()
    pat = r'|'.join(vc[vc > N].index)
    merged_df = nonan_df[nonan_df['industry'].str.contains(pat)]

    return merged_df


def text_preprocess(merged_df):
    # word segmentation + extract keywords (using jieba)
    # Your code here
    # Answer begin

    business_scope = merged_df['business_scope']
    words = []
    tags = []
    for i in business_scope.index:
        words.append(" ".join(jieba.cut(business_scope[i], cut_all=True)))
        tags.append(" ".join(jieba.analyse.extract_tags(
            business_scope[i], topK=50)))
    merged_df['business_scope'] = tags
    merged_df['business_scope'] += words

    return merged_df

    # Answer end
