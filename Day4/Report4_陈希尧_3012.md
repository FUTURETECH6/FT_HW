<center style = "font-size: 4em">金融科技导论实验报告</center><br/><br/><br/><br/>

**姓名**：<u>陈希尧</u>

**学号**：<u>3180103012</u>

**专业**：<u>计算机科学与技术</u>

**课程名称**：<u>金融科技导论</u>

<center style = "font-size: 1.7em">Table of Contents</center>

[TOC]

## 实验目的

本实验旨在对金融文本数据进行自然语言处理并对结果进行评估分析。

具体内容为对TUSHARE 上市公司基本信息中的股票经营范围文本数据进行自然语言处理，并根据相应的行业分类标签进行文本分类，并评估分类结果。

## 实验步骤

### 获取数据

根据经营范围文本数据接口说明与示例( https://tushare.pro/document/2?docid=112 ) 获得经营范围文本数据，其中 fields 参数选择股票代码“ts_code”和经营范围 “business_scope”。 

然后去掉空值。

```python
df0 = pro.stock_company(exchange='SZSE', fields='ts_code, business_scope')
df1 = df0.dropna(axis=0, how='any')
```

根据行业分类标签数据接口说明与示例(http://tushare.org/classifying.html)获得行业分类标签数据。根据股票代码“ts_code”，可将上一步中获得的股票的经营范围与行业名称相对应。

```python
df2 = pro.stock_basic(exchange='SZSE', fields='ts_code, name, industry')
```

由于tushare的接口进行过更新，这里是不需要进行额外处理即可合并的：

```python
df = pd.merge(df1, df2, how='right')
nonan_df = df.dropna(axis=0, how='any')
vc  = nonan_df['industry'].value_counts()
pat = r'|'.join(vc[vc>N].index)          
merged_df  = nonan_df[nonan_df['industry'].str.contains(pat)]
```

### 将文本数据数值化

1. 根据官网说明进行“结巴”中文分词的安装。 “结巴”中文分词官方参考文档:(https://github.com/fxsjy/jieba)。 
2. 利用“结巴”中文分词技术对经营范围文本数据进行分词。 
3. 利用“结巴”中文分词技术对经营范围文本数据进行关键词提取。 
4. 分词结果和关键词串联作为预处理后的文本数据。 
5. 对预处理后的文本数据进行词频向量化，并进行 TF-IDF 处理得到文本数据数值化向量。 


利用jieba对经营范围文本数据进行分词和关键词提取（中间需要以'\\s'隔开，便于之后的识别），然后再将结果和关键词串联作为预处理的文本

```python
business_scope = merged_df['business_scope']
words = []
tags = []
for i in business_scope.index:
    words.append(" ".join(jieba.cut(business_scope[i], cut_all = True)))
    tags.append(" ".join(jieba.analyse.extract_tags(business_scope[i], topK=50)))
merged_df['business_scope'] = tags
merged_df['business_scope'] += words
```

### 基于数值化文本向量进行分类器学习

1. 进行训练集和测试集的划分。参考工具:sklearn KFold
2. 构建朴素贝叶斯多项式分类器。由于行业标签数量众多，可筛选出单类数据量大于 80 的类进行学习。
    分类器参考工具:sklearn MultinomialNB。
3. 对分类器的效果进行评估，评价指标为 precision，recall，F1-score。 分类评价参考工具:sklearn classification_report

用KFold进行数据集和测试机的划分

```python
kf = KFold(n_splits=5, shuffle=True, random_state=2)
split_result = next(kf.split(processed_df), None)
train = processed_df.iloc[split_result[0]]
test = processed_df.iloc[split_result[1]]
```

构建朴素贝叶斯多项式分类器

```python
clf = MultinomialNB()
clf.fit(X_train_tf, train['industry'])
y_predict = clf.predict(X_test_tf)
```

## 实验结果与分析

### 参数的影响

#### 是否串联

是否将文本分词和关键词串联会影响到之后分类器的行为，此处我尝试在不同的min_df的条件下测试这个的影响(topK=50, n_split=5, cut_all=True)

| min_df             | 1    | 0.1  | 0.01 |
| ------------------ | ---- | ---- | ---- |
| 只用关键词的F1     | 0.59 | 0.44 | 0.60 |
| 串联关键词和分词F1 | 0.55 | 0.58 | 0.62 |

由结果见得，当min_df较大时，只使用关键词能够获得更好的分类效果，随着min_df的减小，串联关键词和分词能大幅提高分类效果，但当min_df足够小时，串联的效果又变得十分有限。

推测是因为若min_df较大，可能会高于很多词的词频，此时若串联关键词和分词，会导致词频进一步下降（词数增加了），因此导致某些目标词被排除。而当min_df减小时，很多重要词的词频已经高于min_df了，此时词数增加也不会影响重要词，反倒会增加重要词的个数（文本中可能有没被提取的关键词），这样就会改进实际的分类效果。

#### min_df

笔者一开始在测试时，发现“通信设备”这一块的precision和recall始终是0，调试发现predict里面根本没有通讯设备，进一步调试发现由于通讯设备的词频较低，需要在向量化时使用更低的下阈值。

以下是在topK=50, n_split=5, cut_all=True, 无串联的情况下进行的测试：

| min_df               | 1              | 0.5            | 0.1            | 0.05           | 0.01           |
| -------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| 通信设备：pre/rec/F1 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/0.11/0.19 | 0.00/0.00/0.00 |
| 平均水平：pre/rec/F1 | 0.66/0.64/0.59 | 0.31/0.35/0.24 | 0.44/0.51/0.44 | 0.68/0.60/0.55 | 0.60/0.66/0.60 |

可以看到仅在min_df为0.05时通信设备的数据有被部分（仅11%）正确分类，这说明了通信设备类的文本信息极易和其他类型的混淆。

对于总效果的影响来看，当min_df小于1时，基本上效果是随着其减小而变好的。

#### n_split

这是表示训练集和测试机分配的参数，即n-1份的训练集和1份的测试集，由于默认的10存在过拟合的现象，因此我对这个参数也进行了测试(topK=50, min_df=0.01, cut_all=True, 串联)

| n_split       | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| avg F1        | 0.60 | 0.62 | 0.60 | 0.62 | 0.58 | 0.58 | 0.55 | 0.54 | 0.54 |
| total Support | 350  | 233  | 175  | 140  | 117  | 100  | 88   | 78   | 70   |

可见当n_split过大时，F1会减小，同时由于比例的改变，total support也会减小

#### topK

topK是修改关键词的个数的参数(n_split=5, min_df=0.01, cut_all=True, 串联)

| topK   | 5    | 20   | 50   | 100  |
| ------ | ---- | ---- | ---- | ---- |
| avg F1 | 0.61 | 0.60 | 0.62 | 0.62 |

可见，在串联过关键词和分词的情况下，提升关键词个数的对提升效果的作用是微乎其微的。

#### cut_all

cut_all参数是调节分词时是否将所有能够分割的词都分割开来的，举例来说，在cut_all关闭的情况下`经营各类商品和技术的进出口`会被分割成`经营 各类 商品 和 技术 的 进出口 `，而打开cut_all则会被分割成`经营 各类 商品 和 技术 的 进出 进出口 出口`

在n_split=5, min_df=0.01, topK=50, 串联的条件下进行测试

| cut_all | False | True |
| ------- | ----- | ---- |
| avg F1  | 0.59  | 0.62 |

可以发现开启cut_all的效果要略好一些

### 最终结果

在串联分词与关键词、min_df=0.01、n_split=5、topK=50、cut_all=True的条件下，我获得的结果如下：

|             | precision | recall | f1-score | support |
| ----------- | --------- | ------ | -------- | ------- |
| 专用机械    | 1.00      | 0.38   | 0.56     | 13      |
| 元器件      | 0.73      | 0.79   | 0.76     | 34      |
| 化工原料    | 0.75      | 0.90   | 0.82     | 20      |
| 电气设备    | 0.80      | 0.57   | 0.67     | 21      |
| 软件服务    | 0.56      | 1.00   | 0.72     | 33      |
| 通信设备    | 0.00      | 0.00   | 0.00     | 19      |
| avg / total | 0.63      | 0.68   | 0.62     | 140     |

其中，通信设备和专用机械类的F1值较低，主要是因为数据来源中这两个行业中很多上市公司的`business_scope`都使用了非常模棱两可的用词，以至于会和其他行业混淆（在训练时就无法区分，更别说测试了），但其他行业的分类效果都还不错。

## 附录

### get_data

```python
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
```

### text_preprocess

```python
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
```

### TF_IDF

```python
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
```

### classification

```python
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
```

