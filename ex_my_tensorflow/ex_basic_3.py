import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

def simple_dcnn(x,name = 'y') : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('dcnn') :
        
        conv1 = tf.layers.conv2d(x,32,[3,3], strides=(1,1),padding = 'valid')
        r1 = tf.nn.elu(conv1)#32*26*26
        
        conv2 = tf.layers.conv2d(r1,64,[3,3], strides=(1,1),padding = 'valid')
        r2 = tf.nn.elu(conv2)#64*24*24
        
        conv3 = tf.layers.conv2d(r2,128,[3,3], strides=(1,1),padding = 'valid')
        r3 = tf.nn.elu(conv3)#128*22*22
        
        conv4 = tf.layers.conv2d(r3,256,[3,3], strides=(1,1),padding = 'valid')
        r4 = tf.nn.elu(conv4)#256*20*20
        
        conv5 = tf.layers.conv2d_transpose(r4,128,[3,3], strides=(1,1),padding = 'valid')
        r5 = tf.nn.elu(conv5)#128*22*22
        
        conv6 = tf.layers.conv2d_transpose(r5,64,[3,3], strides=(1,1),padding = 'valid')
        r6 = tf.nn.elu(conv6)#64*24*24
  
        conv7 = tf.layers.conv2d_transpose(r6,32,[3,3], strides=(1,1),padding = 'valid')
        r7 = tf.nn.elu(conv7)#32*26*26

        conv8 = tf.layers.conv2d_transpose(r7,1,[3,3], strides=(1,1),padding = 'valid')
    r8 = tf.nn.sigmoid(conv8,name=name)#1*28*28


    return r8


with tf.device('/gpu:0') :
    u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
    t = tf.placeholder(tf.float32, shape = (None, 28,28,1), name='t')
    y = simple_dcnn(u,name='y')
    loss = tf.reduce_mean(0.5*(-t*tf.log(y + 1e-8) - (1-t)*tf.log(1-y + 1e-8)),name='loss')
    optim = tf.train.AdamOptimizer(0.0001).minimize(loss,name='optim')
    



sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

test_origin,_ = mnist.test.next_batch(16)    
test_input = np.minimum(test_origin +  np.random.uniform(size = (16,784)), 1.0)
  
for i in range(2001) :
    
    train_origin,_ = mnist.train.next_batch(100)
    
    train_input = np.minimum(train_origin +  np.random.uniform(size = (100,784)), 1.0)
    
    _ , e = sess.run([ optim, loss],
            feed_dict={u : np.reshape(train_input,(-1,28,28,1)), t : np.reshape(train_origin,(-1,28,28,1))})
    
    if i%1000 == 0:
        print('e : %.8f'%(e))

        r = sess.run([y],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            t : np.reshape(test_origin,(-1,28,28,1))})
   
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))
      
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')









