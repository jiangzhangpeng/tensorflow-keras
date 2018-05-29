# encoding:utf-8
import tensorflow as tf


def test1():
    ts_c = tf.constant(2, name='ts_c')
    ts_x = tf.Variable(ts_c + 5, name='ts_x')

    print(ts_c)
    print(ts_x)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('ts_c = ', sess.run(ts_c))
        print('ts_x = ', sess.run(ts_x))
        print(ts_c.eval(session=sess))


def test2():
    width = tf.placeholder('int32')
    height = tf.placeholder('int32')
    area = tf.multiply(width, height)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('area = ', sess.run(area, feed_dict={width: 6, height: 5}))


def test3():
    ts_x = tf.Variable([1, 2, 3])
    ts_y = tf.Variable([[1, 2, 1], [3, 4, 3]])
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        X = sess.run(ts_x)
        Y = sess.run(ts_y)
        print(X)
        print(Y)
        print(X.shape)
        print(Y.shape)


def test4():
    x = tf.Variable([[1, 2, 3]])
    w = tf.Variable([[1, 2], [3, 4], [5, 6]])
    xw = tf.matmul(x, w)
    b = tf.Variable([[5,10]])
    xwb = xw + b

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(xw))
        print(sess.run(xwb))
        print(xwb.eval(session=sess))


if __name__ == '__main__':
    test4()
