# encoding:utf-8
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def plot_acc_loss(acc, loss):
    plt.plot(acc)
    plt.plot(loss)
    plt.title('train history')
    plt.ylabel('acc\loss')
    plt.xlabel('epochs')
    plt.legend(['acc', 'loss'], loc='upper left')
    plt.show()


def load_data():
    mnist = input_data.read_data_sets('mnist/', one_hot=True)
    print(len(mnist.train.labels))
    return mnist


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')


def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # SAME注意区分大小写


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_model():
    with tf.name_scope('Input_Layer'):
        x = tf.placeholder('float', shape=[None, 784], name='x')
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('C1_Conv'):
        W1 = weight([3, 3, 1, 16])
        b1 = bias([16])
        Conv1 = conv2D(x_image, W1) + b1
        C1_Conv = tf.nn.relu(Conv1)

    with tf.name_scope('C1_Pool'):
        C1_Pool = max_pool_2x2(C1_Conv)

    with tf.name_scope('C2_Conv'):
        W2 = weight([3, 3, 16, 36])
        b2 = bias([36])
        Conv2 = conv2D(C1_Pool, W2) + b2
        C2_Conv = tf.nn.relu(Conv2)

    with tf.name_scope('C1_Pool'):
        C2_Pool = max_pool_2x2(C2_Conv)

    with tf.name_scope('D_Flat'):
        D_Flat = tf.reshape(C2_Pool, [-1, 7 * 7 * 36])

    with tf.name_scope('D_Hidden_Layer'):
        W3 = weight([7 * 7 * 36, 128])
        b3 = bias([128])
        D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3) + b3)
        D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob=0.8)

    with tf.name_scope('Output_Layer'):
        W4 = weight([128, 10])
        b4 = bias([10])
        y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4) + b4)

    with tf.name_scope('optimizer'):
        y_label = tf.placeholder('float', shape=[None, 10], name='y_label')
        loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

    with tf.name_scope('evaluate_model'):
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    return x, y_label, optimizer, loss_function, accuracy


def train(mnist, x, y_label, optimizer, loss_function, accuracy):
    train_epochs = 15
    batch_size = 550
    total_batches = int(mnist.train.num_examples / batch_size)
    epoch_list = []
    accuracy_list = []
    loss_list = []
    start_time = time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(train_epochs):
            for i in range(total_batches):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})
                # print('epoch: %d, bathes: %d' % (epoch, i))
            loss, acc = sess.run([loss_function, accuracy],
                                 feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels})
            print('train epoch: %02d , loss: %.5f , accuracy: %.5f' % (epoch, loss, acc))
            epoch_list.append(epoch)
            accuracy_list.append(acc)
            loss_list.append(loss)
        print('test accuracy', sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('D:/WorkSoft/Python/Python3.6.4/Scripts/log/mnist_cnn', sess.graph)
    duration = time() - start_time
    print('train finished takes :', duration)
    plot_acc_loss(accuracy_list, loss_list)


if __name__ == '__main__':
    mnist = load_data()
    x, y_label, optimizer, loss_function, accuracy = create_model()
    train(mnist, x, y_label, optimizer, loss_function, accuracy)
