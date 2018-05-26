# encoding:utf-8
import os
import urllib
import pandas as pd
import numpy as np
from sklearn import preprocessing


def get_data():
    url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
    file_path = 'titanic3.xls'
    if not os.path.isfile(file_path):
        result = urllib.request.urlretrieve(url, file_path)
        print('downloaded:', result)
    all_df = pd.read_excel(file_path)
    cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    all_df = all_df[cols]
    msk = np.random.rand(len(all_df)) < 0.8
    train_df = all_df[msk]
    test_df = all_df[~msk]
    return train_df, test_df


def pre_process_data(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    x_one_hot_df = pd.get_dummies(data=df, columns=['embarked'])
    nd_array = x_one_hot_df.values
    features = nd_array[:, 1:]
    label = nd_array[:, 0]
    min_max_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_features = min_max_scale.fit_transform(features)
    return scaled_features, label


if __name__ == '__main__':
    train_df, _ = get_data()
    train_features, train_label = pre_process_data(train_df)
    print(train_features[:2])
    print(train_label[:2])
