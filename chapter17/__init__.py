# encoding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.show()


def plot_image_labels_prediction(images, labels, predictions, idx, num=10):
    fig = plt.figure()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(np.reshape(images[idx], (28, 28)), cmap='binary')
        title = 'label = ' + str(np.argmax(labels[idx]))
        if len(predictions) > 0:
            title += ',prediction = ' + str(predictions[idx])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


if __name__ == '__main__':
    mnist = input_data.read_data_sets('mnist/', one_hot=True)
    plot_image_labels_prediction(mnist.train.images, mnist.train.labels, [], 0, 25)
