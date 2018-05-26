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



