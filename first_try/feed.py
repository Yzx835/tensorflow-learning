import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
output2 = tf.add(input1, input2)

with tf.Session() as sess:
    print(sess.run([output, output2], feed_dict={input1: [7.], input2: [2.]}))
