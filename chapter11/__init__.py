# encoding:utf-8

import urllib.request
import os
from sklearn import preprocessing
import numpy
import pandas as pd

url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
filepath = 'titanic3.xls'
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)

all_df = pd.read_excel(filepath)
# print(all_df.describe())

cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]
# print(all_df[:2])

df = all_df.drop(['name'], axis=1)
# print(df[:2])

# 空缺值填充
age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)

# 字符数值转换
df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)

# 非数值变量onehot编码
x_Onehot_df = pd.get_dummies(data=df, columns=['embarked'])

# print(df.isnull().sum())
# print(x_Onehot_df[:5])
ndarray = x_Onehot_df.values
# print(ndarray.shape)
# 分离feature和label
label = ndarray[:, 0]
features = ndarray[:, 1:]
# print(label.shape)
# print(features.shape)


# 数据标准化  数值数据整理为[0,1]区间


minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaledFeature = minmax_scale.fit_transform(features)
# print(scaledFeature[:2])
# print(features[:2])


msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
#print(msk)
#print(~msk)
