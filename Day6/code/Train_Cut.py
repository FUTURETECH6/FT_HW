import pandas as pd

data = pd.read_csv("../Day2/data/train_new.csv")
data.sample(frac=1.)
data.reset_index(drop=True)

# data.dropna(inplace=True)

data.dropna(inplace=True, thresh=1)
for i in data.columns:
    data[i].fillna(data[i].mean(), inplace=True)

num_train = int(data.shape[0] * 0.8)
train_data = data[:num_train]
test_data = data[num_train:]
print(train_data.shape, test_data.shape, sep='\n')
train_data.to_csv("./data/train.csv")
test_data.to_csv("./data/test.csv")