import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)





X = tf.placeholder(tf.float32, shape=[None, 4])

W = tf.Variable(tf.random_normal(shape = [4,2], mean = 0,stddev = 0.1,seed = 1))
b = tf.Variable(tf.random_normal(shape = [2], mean = 0,stddev = 0.1,seed = 2))

Y =tf.matmul(X,W) + b

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())



para = [W,b]

para1 = sess.run([W,b])
print(para1)


_,result_para = sess.run([Y,para],feed_dict={X : [[1,1,1,1]]})


print(result_para)





