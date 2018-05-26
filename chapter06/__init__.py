# encoding:utf-8

import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
from numpy import shape
import matplotlib.pyplot as plt

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data('D:\\\Workspaces\\tensorflow-keras\\chapter06\mnist.npz')
print(shape(x_train_image))


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    plt.imshow(image, cmap='binary')
    plt.show()

    
def test1():
    print(y_train_label[0])
    plot_image(x_train_image[0])
    
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25:
        num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(images[idx],cmap='binary')
        title = 'label=' + str(labels[idx])
        if len(prediction) > 0:
            title += ',prediction=' + str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

def test2():
    print(x_test_image[0])
    plot_images_labels_prediction(x_test_image,y_test_label,[],0,30)
    
    
def preprocessX(x_train_image,x_test_image):
    print('x_train_image shape:',x_train_image.shape)
    print('x_test_image shape:',x_test_image.shape)
    x_train = x_train_image.reshape(60000,28*28).astype('float32')
    print('x_train shape',x_train.shape)
    x_test = x_test_image.reshape(10000,28*28).astype('float32')
    print('x_test shape:',x_test.shape)
    x_train_norm = x_train / 255
    x_test_norm = x_test / 255
    print(x_test[0])
    return x_train_norm,x_test_norm
    


def test3():
    preprocessX(x_train_image, x_test_image)
    
    
def preprocessY(y_train_label,y_test_label):
    print(y_train_label[:5])
    
def test4():
    preprocessY(y_train_label, y_test_label)
    y_train_onehot = np_utils.to_categorical(y_train_label, 10)
    y_test_onehot = np_utils.to_categorical(y_test_label,10)
    print(y_train_onehot[:5])
    return y_train_onehot,y_test_onehot


if __name__ == '__main__':
    
    test4()
    
