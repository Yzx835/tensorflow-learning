import input_data
import tensorflow as tf
import numpy as np
import os
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
"""
print(mnist.train.labels.shape)
print(mnist.train.images.shape)
print(mnist.validation.labels.shape)
print(mnist.validation.images.shape)
print(mnist.test.labels.shape)
print(mnist.test.images.shape)
"""


x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

saver = tf.train.Saver()
saver.save(sess, "train-result/train", global_step=1000)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
