import tensorflow as tf
import numpy as np
import os
import sys
import time
import my_lib 
   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)


def simple_G(x, reuse = False, name = 'G_out') : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('G',reuse=reuse) :
        
        #x = (-1, 1, 1, 100)
        conv1 = tf.layers.conv2d_transpose(x,1024,[4,4], strides=(1,1),padding = 'valid') 
        r1 = tf.nn.elu(conv1)#1024*4*4
        
        conv2 = tf.layers.conv2d_transpose(r1,512,[4,4], strides=(2,2),padding = 'same')
        r2 = tf.nn.elu(conv2)#512*8*8
        
        conv3 = tf.layers.conv2d_transpose(r2,256,[4,4], strides=(2,2),padding = 'same')
        r3 = tf.nn.elu(conv3)#256*16*16

        conv4 = tf.layers.conv2d_transpose(r3,128,[4,4], strides=(2,2),padding = 'same')
        r4 = tf.nn.elu(conv4)#128*32*32

        conv5 = tf.layers.conv2d(r4,64,[3,3], strides=(1,1),padding = 'valid')
        r5 = tf.nn.elu(conv5)#64*30*30

        conv6 = tf.layers.conv2d(r5,1,[3,3], strides=(1,1),padding = 'valid')
        r6 = tf.nn.sigmoid(conv6,name=name)#1*28*28
  

        return r6 

def simple_D(x,reuse = False) :
    
    with tf.variable_scope('D', reuse=reuse) :
        
        #x = (-1,28,28,1)
        conv1 = tf.layers.conv2d(x,64,[5,5], strides=(1,1),padding = 'valid')
        r1 = tf.nn.elu(conv1)#64*24*24

   
        conv2 = tf.layers.conv2d(r1,128,[5,5], strides=(1,1),padding = 'valid')
        r2 = tf.nn.elu(conv2)#128*20*20

  
        conv3 = tf.layers.conv2d(r2,256,[5,5], strides=(1,1),padding = 'valid')
        r3 = tf.nn.elu(conv3)#256*16*16

 
        conv4 = tf.layers.conv2d(r3,512,[4,4], strides=(2,2),padding = 'same')
        r4 = tf.nn.elu(conv4)#512*8*8


        conv5 = tf.layers.conv2d(r4,1024,[4,4], strides=(2,2),padding = 'same')
        r5 = tf.nn.elu(conv5)#1024*4*4

       
        conv6 = tf.layers.conv2d(r5,1,[4,4], strides=(1,1),padding = 'valid')
        r6 = tf.nn.sigmoid(conv6)#1*1*1


        return r6



with tf.device('/gpu:1') :

    z = tf.placeholder(tf.float32,shape=(None,1,1,100),name = 'z')    
    u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
    
    G_z = simple_G(z,name='G_z')

    D_real = simple_D(u)
    D_fake = simple_D(G_z,reuse=True)

    D_loss =  tf.reduce_mean(-0.5*(tf.log(D_real + 1e-8) + tf.log(1-D_fake + 1e-8)),name='D_loss')
    G_loss =  tf.reduce_mean(-0.5*(tf.log(D_fake + 1e-8)),name='G_loss')
 

    #D_loss =  tf.reduce_mean(0.5*(tf.square(D_real - 1) + tf.square(D_fake)),name='D_loss')
    #G_loss =  tf.reduce_mean(0.5*tf.square(D_fake -1),name='G_loss')
 

    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('D')]
    G_vars = [var for var in T_vars if var.name.startswith('G')]


    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :    
        D_optim = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list=D_vars, name='D_optim')
     
        G_optim = tf.train.AdamOptimizer(0.00001).minimize(G_loss, var_list=G_vars, name='G_optim')




sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

np.random.seed(int(time.time()))


test = np.random.normal(0,1,size=(16,1,1,100))


for i in range(10000) :
    
    train_images,_ = mnist.train.next_batch(100)
    latent_val = np.random.normal(0,1,size=(100,1,1,100))
    _ , D_e = sess.run([ D_optim, D_loss], feed_dict={u : np.reshape(train_images,(-1,28,28,1)), z : latent_val})
    

    while True :
        
        latent_val = np.random.normal(0,1,size=(100,1,1,100)) 
        _ , G_e = sess.run([ G_optim, G_loss], feed_dict={u : np.reshape(train_images,(-1,28,28,1)), z : latent_val})
        if G_e < 1 :
            break;



    if i%100 == 0:
        print('D_e : %.8f, G_e : %.8f'%(D_e,G_e))
    if i%100 == 0 :
        r = sess.run([G_z],feed_dict={z : test})
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))
   
      
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')









