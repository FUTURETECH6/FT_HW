import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"../Day2/data/train_new.csv").drop(columns=['id'])

data.corr()['Y'].where(data.corr()['Y'] > 0.1).dropna()

sns.catplot(x='VAR_NAME', y='IV', data=IV[i for i in ], aspect=4)