import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

"""
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
"""

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        result = sess.run([product])
        print(result)
