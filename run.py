from IO.Read import read_csv_dataset
import pandas as pd
import numpy as np

dataset = read_csv_dataset("./dataset/numerical_dataset.csv")

perms = np.random.permutation(len(dataset))



train_X = dataset.iloc[perms[:int(.7 * len(perms))]].drop('word', axis=1)
train_Y = train_X['keyword'].replace((True, False), (1, 0))
train_X.drop('keyword', axis=1, inplace=True)

test_X = dataset.iloc[perms[int(.7 * len(perms)):]].drop('word', axis=1)
test_Y = test_X['keyword'].replace((True, False), (1, 0))
test_X.drop('keyword', axis=1, inplace=True)

print()