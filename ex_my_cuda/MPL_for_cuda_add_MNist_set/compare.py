import tensorflow as tf
import numpy as np
import os
   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)

mini_batch_size = 128
max_epoch = 10000
learning_rate = 0.2


X = tf.placeholder(tf.float32, shape=[None, 784])
T = tf.placeholder(tf.float32, shape=[None, 10])



W1 = tf.Variable(tf.random_normal(shape = [784, 512], mean = 0,stddev = 0.1))
b1 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.001))
W2 = tf.Variable(tf.random_normal(shape = [512, 128], mean = 0,stddev = 0.1))
b2 = tf.Variable(tf.random_normal(shape = [128], mean = 0,stddev = 0.001))
W3 = tf.Variable(tf.random_normal(shape = [128, 10], mean = 0,stddev = 0.1))
b3 = tf.Variable(tf.random_normal(shape = [10], mean = 0,stddev = 0.001))



a2 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
a3 = tf.nn.sigmoid(tf.matmul(a2, W2) + b2)
a4 = tf.nn.sigmoid(tf.matmul(a3, W3) + b3)
Y = a4 




G_loss = -tf.reduce_mean(T*tf.log(Y) + (1-T)*tf.log(1-Y + 0.000000001) )

G_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(G_loss)

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())



        
for it in range(max_epoch) :
   

    train_input, train_label = mnist.train.next_batch(mini_batch_size)
    



    _, G_loss_curr = sess.run([G_solver, G_loss],feed_dict={X : train_input, T: train_label})







