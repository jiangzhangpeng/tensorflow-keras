# encoding:utf-8
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.show()


def plot_acc_loss(acc, loss):
    plt.plot(acc)
    plt.plot(loss)
    plt.title('train history')
    plt.ylabel('acc\loss')
    plt.xlabel('epochs')
    plt.legend(['acc', 'loss'], loc='upper left')
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


def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


def train_data():
    mnist = input_data.read_data_sets('mnist/', one_hot=True)
    # plot_image_labels_prediction(mnist.train.images, mnist.train.labels, [], 0, 25)

    # 定义输入层，隐藏层，输出层
    x = tf.placeholder('float', [None, 28 * 28])
    h1 = layer(output_dim=1000, input_dim=784, inputs=x, activation=tf.nn.relu)
    h2 = layer(output_dim=1000, input_dim=1000, inputs=h1, activation=tf.nn.relu)
    y_predict = layer(output_dim=10, input_dim=1000, inputs=h2, activation=None)
    # 定义label
    y_label = tf.placeholder('float', [None, 10])
    # 定义损失函数
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    # 定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

    # 比对预测结果
    correct_predict = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))

    # 定义训练参数
    train_epochs = 15
    batch_size = 100
    total_batchs = 550
    loss_list = []
    epoch_list = []
    accuracy_list = []

    start_time = time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(train_epochs):
            for i in range(total_batchs):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})
            loss, acc = sess.run([loss_function, accuracy],
                                 feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels})

            epoch_list.append(epoch)
            loss_list.append(loss)
            accuracy_list.append(acc)
            print('train epoch: %02d , loss: %.5f , accuracy: %.5f' % (epoch, loss, acc))
        print('test accuracy', sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
    duration = time() - start_time
    print('train finished takes :', duration)
    plot_acc_loss(accuracy_list, loss_list)


if __name__ == '__main__':
    train_data()
