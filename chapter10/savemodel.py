#encoding:utf-8
'''
Created on 2018��5��22��

@author: chong
'''

# encoding:utf-8

from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
np.random.seed(10)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D

def preparedata():
    # keras该方法写的比较烂 凑合用吧
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # 标准化
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0
    # label one hot
    y_train_onehot = np_utils.to_categorical(y_train, 10)
    y_test_onehot = np_utils.to_categorical(y_test, 10)
    
    return (x_train_norm,y_train_onehot),(x_test_norm,y_test_onehot)
    


label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

def getmodel():
    # 定义模型
    model = Sequential()
    # 卷积层1
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # 池化层1
    model.add(MaxPooling2D((2, 2)))
    
    
    #卷积层2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # 池化层2
    model.add(MaxPooling2D((2, 2)))
    
        
    #卷积层3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # 池化层3
    model.add(MaxPooling2D((2, 2)))
    
    
    #平滑层
    model.add(Flatten())
    model.add(Dropout(0.3))
    
    #fully connect layer
    model.add(Dense(2500,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1500,activation='relu'))
    model.add(Dropout(0.3))
    
    #output layer
    model.add(Dense(10,activation='softmax'))
    
    #定义训练方式
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    

def train():
    (x_train,y_train),(x_test,y_test) = preparedata()
    model = getmodel()
    try:
        model.load_weights('deepmodel.h5')
        print('load success,continue training!')
    except:
        print('load error!')
    history = []
    for i in range(50):
        his = model.fit(x_train,y_train,validation_split=0.2,batch_size=256,epochs=1)
        history.append(his)
        model.save_weights('deepmodel.h5')
        print('epoches ',str(i+1),' saved!')
    
    scores = model.evaluate(x_test, y_test)
    print(scores[1])


if __name__ == '__main__':
    train()
    pass