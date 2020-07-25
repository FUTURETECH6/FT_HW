# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback

data = pd.read_csv(r"../Day2/data/train_new.csv").drop(columns=['id'])

# %% [markdown]
# ## 特征与Label关系分析
# %% [markdown]
# ### 关联性

# %%
corr_map = data.corr()
plt.figure(figsize=(20, 20))
mask = np.triu(np.ones_like(corr_map, dtype=np.bool))
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=12)
sns.heatmap(corr_map, mask=mask, cmap=cmap, square=True, linewidths=.5)
plt.savefig('./assets/corr.svg', format='svg')


# %%
data.corr()['Y'].where(abs(data.corr()['Y']) > 0.12).drop(labels=['Y']).dropna()

# %% [markdown]
# ### WOE与IV指标

# %%
max_bin = 20
force_bin = 3

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


# %%
final_iv, IV = data_vars(data, data.Y)
IV = IV.set_index('VAR_NAME').loc[[('X' + str(i)) for i in range(1, 73)]].reset_index()


# %%
final_iv.head(10)


# %%
plt.figure(figsize=(30,15))
sns.catplot(x='VAR_NAME', y='IV', data=IV, aspect=4)


# %%
IV.where(IV > 0.1).dropna().sort_values(by='IV', ascending=False)

# %% [markdown]
# ## 特征工程
# %% [markdown]
# ### Normalization

# %%
digital_cols = data.dtypes[data.dtypes != 'object'].index
nor_data = data.copy()
nor_data[digital_cols] = nor_data[digital_cols].apply(lambda x: (x - x.mean()) / (x.std()))
nor_data.head(10)


# %%
num_train = int(nor_data.shape[0] * 0.8)
train_data = nor_data[:num_train]
test_data = nor_data[num_train:]
train_data.to_csv('./data/train_nor.csv', sep=',', index=False, header=True)
train_data.to_csv('./data/test_nor.csv', sep=',', index=False, header=True)

# %% [markdown]
# ### WOE

# %%
data.columns.difference(['target'])


# %%
woe_data = data.copy()
woe_data.fillna(-1,inplace=True)
transform_vars_list = woe_data.columns.difference(['target'])
transform_prefix = 'new_'

for var in transform_vars_list:
    small_df = final_iv[final_iv['VAR_NAME'] == var]
    transform_dict = dict(zip(small_df.MAX_VALUE,small_df.WOE))
    replace_cmd = ''
    replace_cmd1 = ''
    for i in sorted(transform_dict.items()):
        replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '
        replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '
    replace_cmd = replace_cmd + '0'
    replace_cmd1 = replace_cmd1 + '0'
    if replace_cmd != '0':
        try:
            woe_data[transform_prefix + var] = woe_data[var].apply(lambda x: eval(replace_cmd))
        except:
            woe_data[transform_prefix + var] = woe_data[var].apply(lambda x: eval(replace_cmd1))


# %%
woe_data.head(10)


# %%
num_train = int(woe_data.shape[0] * 0.8)
train_data = woe_data[:num_train]
test_data = woe_data[num_train:]
train_data.to_csv('./data/train_woe.csv', sep=',', index=False, header=True)
train_data.to_csv('./data/test_woe.csv', sep=',', index=False, header=True)

# %% [markdown]
# ### 特征交叉

# %%
def add_cross_feature(data, feature_1, feature_2):
    comb_index = data[[feature_1, feature_2]].drop_duplicates()
    comb_index[feature_1 + '_' + feature_2] = np.arange(comb_index.shape[0])
    data = pd.merge(data, comb_index, 'left', on=[feature_1, feature_2])
    return data


# %%
cross_data = data.copy()
cross_data.X27 = pd.qcut(cross_data.X27, q=10, duplicates='drop')
cross_data.X30 = pd.qcut(cross_data.X30, q=10, duplicates='drop')
cross_data = add_cross_feature(cross_data, 'X27', 'X30')


# %%
cross_data.head(10)


# %%
num_train = int(cross_data.shape[0] * 0.8)
train_data = cross_data[:num_train]
test_data = cross_data[num_train:]
train_data.to_csv('./data/train_cross.csv', sep=',', index=False, header=True)
train_data.to_csv('./data/test_cross.csv', sep=',', index=False, header=True)


