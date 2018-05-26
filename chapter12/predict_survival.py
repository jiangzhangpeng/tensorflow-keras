# encoding:utf-8
import urllib
from sklearn import preprocessing
import pandas as pd
import os
from keras.models import Sequential, load_model


def get_data():
    url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
    file_path = 'titanic3.xls'
    if not os.path.isfile(file_path):
        result = urllib.request.urlretrieve(url, file_path)
        print('downloaded:', result)
    all_df = pd.read_excel(file_path)
    cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    all_df = all_df[cols]
    Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.000, 'S'])
    Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])
    JR_df = pd.DataFrame([list(Jack), list(Rose)], columns=cols)
    all_df = pd.concat([all_df, JR_df])
    return all_df


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


def get_model():
    model = load_model(filepath='model_save.h5')
    return model


if __name__ == '__main__':
    all_df = get_data()
    features, labels = pre_process_data(all_df)
    model = get_model()
    suvivals = model.predict(features)
    pd = all_df
    pd.insert(len(all_df.columns), 'probability', suvivals)
    print(pd[-2:])

    print(pd[(pd['survived'] == 0) & (pd['probability'] > 0.9)])
