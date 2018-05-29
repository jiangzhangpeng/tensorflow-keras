# encoding:utf-8
import matplotlib.pyplot as plt
import tensorflow as tf


def test1():
    X = tf.Variable([[0.4, 0.2, 0.4]])
    W = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])
    b = tf.Variable([[0.1, 0.2]])
    XWb = tf.matmul(X, W) + b
    y = tf.nn.relu(tf.matmul(X, W) + b)
    y_sigmoid = tf.nn.sigmoid(tf.matmul(X, W) + b)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('XWb: ', sess.run(XWb))
        print('relu y: ', sess.run(y))
        print('sigmoid y: ', sess.run(y_sigmoid))


def test2():
    ts_norm = tf.random_normal([1000])
    with tf.Session() as sess:
        norm_data = ts_norm.eval()
        print(norm_data[0:5])
        plt.hist(norm_data)
        plt.show()


if __name__ == '__main__':
    test2()
