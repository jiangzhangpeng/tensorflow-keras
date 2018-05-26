# encoding:utf-8

from keras.utils import np_utils
import numpy as np
from chapter06 import plot_images_labels_prediction
np.random.seed(10)

from keras.datasets import mnist
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data('D:\\\Workspaces\\tensorflow-keras\\chapter06\mnist.npz')

x_train = x_train_image.reshape(60000, 784).astype('float32')
x_test = x_test_image.reshape(10000, 784).astype('float32')

x_train_normalize = x_train / 255
x_test_normalize = x_test / 255

y_train_onehot = np_utils.to_categorical(y_train_label , 10)
y_test_onehot = np_utils.to_categorical(y_test_label, 10)


from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

#model.add(Dense(units=256,input_dim=784,activation='relu',kernel_initializer='normal'))
#调整为1000隐藏层
model.add(Dense(units=1000,input_dim=784,activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(units=1000,input_dim=784,activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax',kernel_initializer='normal'))

print(model.summary())

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_history = model.fit(x_train_normalize,y_train_onehot,validation_split=0.2,epochs=10,batch_size=256)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validatin'],loc='upper left')
    plt.show()
    
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_test_normalize, y_test_onehot)
print('accuracy = ',scores[1])

#prediction = model.predict_classes(x_test_normalize) 
#两种方式均可以
prediction = model.predict_classes(x_test) 

#plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx = 340)

import pandas as pd
#print(pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['predict']))

df = pd.DataFrame({'label':y_test_label,'predict':prediction})
#print(df[:2])
#print(df[(df.label==5)&(df.predict==3)])


