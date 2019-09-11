import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
import time 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

 
start = time.time()

 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

def simple_dcnn(x,isTrain = True, name = 'y') : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('dcnn') :
        
        conv1 = tf.layers.conv2d(x,64,[3,3], strides=(1,1),padding = 'same') 
        r1 = tf.nn.elu(tf.layers.batch_normalization(conv1,training=isTrain))
        
        conv2 = tf.layers.conv2d(r1,64,[3,3], strides=(1,1),padding = 'same')
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain)) + r1
        
        conv3 = tf.layers.conv2d(r2,64,[3,3], strides=(1,1),padding = 'same')
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain)) + r2

        conv4 = tf.layers.conv2d(r3,64,[3,3], strides=(1,1),padding = 'same')
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain)) + r3

        conv5 = tf.layers.conv2d(r4,64,[3,3], strides=(1,1),padding = 'same')
        r5 = tf.nn.elu(tf.layers.batch_normalization(conv5,training=isTrain)) + r4

        conv6 = tf.layers.conv2d(r5,1,[3,3], strides=(1,1),padding = 'same')
        r6 = tf.nn.elu(tf.layers.batch_normalization(conv6,training=isTrain)) + x
        
        conv7 = tf.layers.conv2d(r6,1,[3,3], strides=(1,1),padding = 'same')
    r7 = tf.nn.sigmoid(conv7,name=name)
   
    return r7


u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
t = tf.placeholder(tf.float32, shape = (None, 28,28,1), name='t')
isTrain = tf.placeholder(dtype=tf.bool,name='isTrain') 
 

y = simple_dcnn(u,isTrain=isTrain,name='y')

loss = tf.reduce_mean(0.5*(-t*tf.log(y + 1e-8) - (1-t)*tf.log(1-y + 1e-8)),name='loss')

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :    
    optim = tf.train.AdamOptimizer(0.0001).minimize(loss,name='optim')
    



sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

np.random.seed(int(time.time()))

test_images,_ = mnist.test.next_batch(16)    
test_origin = test_images*np.random.uniform()

test_input = np.minimum(test_origin + np.random.uniform(size = (16,784)), 1.0)

my_lib.mnist_4by4_save(np.reshape(test_input,(-1,784)),file_name + '/input_noise.png')
my_lib.mnist_4by4_save(np.reshape(test_origin,(-1,784)),file_name + '/ground_true.png') 
 

for i in range(1000000) :
    
    
    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform()
    
    train_input = np.minimum(train_origin + np.random.uniform(size = (100,784)), 1.0)
    
    _ , e = sess.run([ optim, loss],feed_dict={u : np.reshape(train_input,(-1,28,28,1)),
        t : np.reshape(train_origin,(-1,28,28,1)),isTrain : True})
    
    if i%10000 == 0:
        print('e : %.8f'%(e))
        
        r = sess.run([y],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            t : np.reshape(test_origin,(-1,28,28,1)),isTrain : False})
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))
      
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)









