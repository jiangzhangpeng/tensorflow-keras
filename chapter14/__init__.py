# encoding:utf-8
from keras.layers import Dense, Dropout, Flatten, SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from matplotlib import pyplot as plt

import chapter13


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# 建立模型
def create_model():
    model = Sequential()
    # 嵌入层
    # model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
    model.add(Embedding(output_dim=32, input_dim=3800, input_length=380))
    model.add(Dropout(0.4))
    # 平坦层
    model.add(Flatten())
    # 隐藏层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    # 输出层
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# 建立模型
def create_model_RNN():
    model = Sequential()
    # 嵌入层
    # model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
    model.add(Embedding(output_dim=32, input_dim=3800, input_length=380))
    model.add(Dropout(0.4))
    # 平坦层
    model.add(SimpleRNN(16))
    # 隐藏层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    # 输出层
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# 建立模型
def create_model_LSTM():
    model = Sequential()
    # 嵌入层
    # model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
    model.add(Embedding(output_dim=32, input_dim=3800, input_length=380))
    model.add(Dropout(0.4))
    # 平坦层
    model.add(LSTM(32))
    # 隐藏层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    # 输出层
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# 训练模型
def train_model():
    train_labels, train_texts = chapter13.read_files('train')
    test_labels, test_texts = chapter13.read_files('test')
    x_train, x_test = chapter13.text_to_vec(train_texts, test_texts)
    model = create_model_LSTM()
    train_history = model.fit(x_train, train_labels, batch_size=256, epochs=10, validation_split=0.2)
    scores = model.evaluate(x_test, test_labels, batch_size=256)
    print(scores[1])
    model.save('model_save_LSTM32.h5', overwrite=True)
    show_train_history(train_history, 'acc', 'val_acc')
    show_train_history(train_history, 'loss', 'val_loss')
    return model


if __name__ == '__main__':
    model = train_model()
