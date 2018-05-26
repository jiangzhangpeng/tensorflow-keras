# encoding:utf-8

from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
np.random.seed(10)
# keras该方法写的比较烂 凑合用吧
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

   
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx], cmap='binary')
        title = 'label=' + label_dict[labels[idx][0]]
        if len(prediction) > 0:
            title += ',prediction=' + label_dict[prediction[idx]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

# plot_images_labels_prediction(x_train, y_train, [], 0, num=25)


# 标准化
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

# label one hot
y_train_onehot = np_utils.to_categorical(y_train, 10)
y_test_onehot = np_utils.to_categorical(y_test, 10)

# 定义模型
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D

model = Sequential()
# 卷积层1
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
# 池化层1
model.add(MaxPooling2D((2, 2)))
#卷积层2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
# 池化层2
model.add(MaxPooling2D((2, 2)))
#平滑层
model.add(Flatten())
model.add(Dropout(0.25))

#fully connect layer
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))

#output layer
model.add(Dense(10,activation='softmax'))

#定义训练方式
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train
train_history = model.fit(x_train_norm,y_train_onehot,validation_split=0.2,batch_size=256,epochs=10)
print(model.summary())

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validatin'],loc='upper left')
    plt.show()
    
#show_train_history(train_history, 'acc', 'val_acc')
#show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_test_norm, y_test_onehot)
print(scores[1])

    

predictions_classes = model.predict_classes(x_test_norm)
plot_images_labels_prediction(x_test, y_test, predictions_classes, 0, 10)


def show_predicted_probability(y,prediction,x,predicted_probability,i):
    print('label:',label_dict[y[i][0]],'predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_test[i],(32,32,3)))
    plt.show()
    for j in range(10):
        print(label_dict[j]+' probability:%1.9f'%(predicted_probability[i][j]))

predictions = model.predict(x_test_norm)
show_predicted_probability(y_test, predictions_classes, x_test, predictions, 0)


