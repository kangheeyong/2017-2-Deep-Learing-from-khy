import tensorflow as tf
import numpy as np
import os
import time


start = time.time()


   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)


def simple_resnn(x) :
            
    w_init = tf.truncated_normal_initializer(mean=0.0, stddev = 0.02)
    b_init = tf.constant_initializer(0.0)

    # 1st layer
    w1 = tf.get_variable('w1',[x.get_shape()[1],2048],initializer = w_init)
    b1 = tf.get_variable('b1',[2048],initializer = b_init)
    h1 = tf.nn.elu(tf.matmul(x,w1)+b1)


    # 2st layer
    w2 = tf.get_variable('w2',[h1.get_shape()[1],2048],initializer = w_init)
    b2 = tf.get_variable('b2',[2048],initializer = b_init)
    h2 = tf.nn.elu(tf.matmul(h1,w2)+b2) + h1
    

    # 3st layer
    w3 = tf.get_variable('w3',[h2.get_shape()[1],2048],initializer = w_init)
    b3 = tf.get_variable('b3',[2048],initializer = b_init)
    h3 = tf.nn.elu(tf.matmul(h2,w3)+b3) + h2

    # 4st layer
    w4 = tf.get_variable('w4',[h3.get_shape()[1],2048],initializer = w_init)
    b4 = tf.get_variable('b4',[2048],initializer = b_init)
    h4 = tf.nn.elu(tf.matmul(h3,w4)+b4) + h3

    # 5st layer
    w5 = tf.get_variable('w5',[h4.get_shape()[1],2048],initializer = w_init)
    b5 = tf.get_variable('b5',[2048],initializer = b_init)
    h5 = tf.nn.elu(tf.matmul(h4,w5)+b5) + h4

    # 6st layer
    w6 = tf.get_variable('w6',[h5.get_shape()[1],1568],initializer = w_init)
    b6 = tf.get_variable('b6',[1568],initializer = b_init)
    h6 = tf.nn.elu(tf.matmul(h5,w6)+b6) + x

    # 7st layer
    w7 = tf.get_variable('w7',[h6.get_shape()[1],784],initializer = w_init)
    b7 = tf.get_variable('b7',[784],initializer = b_init)
    h7 = tf.nn.sigmoid(tf.matmul(h6,w7)+b7)

    return h7

with tf.device('/gpu:1') :
    u = tf.placeholder(tf.float32, shape = (None, 1568),name='u')
    t = tf.placeholder(tf.float32, shape = (None, 784), name='t')
    y = simple_resnn(u)

    loss = 0.5*tf.reduce_mean(-t*tf.log(y + 1e-8) - (1-t)*tf.log(1-y + 1e-8),name='loss')
    optim = tf.train.AdamOptimizer(0.0001).minimize(loss,name='optim')
    





sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())


for i in range(100000) :
    
    train_origin,_ = mnist.train.next_batch(100)
     
    train_ref,_ = mnist.train.next_batch(100)
    
    temp = np.minimum(train_origin + train_ref, 1.0)
    
    train_input = np.concatenate((temp, train_ref),axis=1)
    _ , e = sess.run([ optim, loss],feed_dict={u : train_input, t : train_origin})
    
    if i%10000 == 0:
        print('e : %.8f'%(e))




end = time.time()-start

print("total time : ",end)


