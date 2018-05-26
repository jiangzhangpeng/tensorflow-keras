# encoding:utf-8

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
np.random.seed(10)
from matplotlib import pyplot as plt

# 读取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data('D:\\\Workspaces\\tensorflow-keras\\chapter06\mnist.npz')

# reshape
x_train4d = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test4d = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalization
x_train4d_norm = x_train4d / 255
x_test4d_norm = x_test4d / 255
print(x_train4d_norm.shape)
# one-hot
y_train_onehot = np_utils.to_categorical(y_train, 10)
y_test_onehot = np_utils.to_categorical(y_test, 10)

# 建立模型
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary())

# 定义训练方式
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 开始训练
train_history = model.fit(x=x_train4d_norm, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=256)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validatin'], loc='upper left')
    plt.show()


show_train_history(train_history, 'acc', 'val_acc')

