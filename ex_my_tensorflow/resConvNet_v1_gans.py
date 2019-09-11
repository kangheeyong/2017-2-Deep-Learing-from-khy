import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
import time 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

 
start = time.time()

 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

def simple_G(x,ref,isTrain = True, name = 'y') : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('G') :
        
        conv1_u = tf.layers.conv2d(x,64,[3,3], strides=(1,1),padding = 'same') 
        conv1_ref = tf.layers.conv2d(ref,64,[3,3], strides=(1,1),padding = 'same') 
        conv1 = tf.layers.batch_normalization(conv1_ref+conv1_u,training=isTrain)
        
        r1 = tf.nn.elu(conv1)
    
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


def simple_D(x,isTrain=True,reuse = False) :
    
    with tf.variable_scope('D', reuse=reuse) :
        
        #x = (-1,28,28,1)
        conv1 = tf.layers.conv2d(x,64,[5,5], strides=(1,1),padding = 'valid')
        r1 = tf.nn.elu(tf.layers.batch_normalization(conv1,training=isTrain))#64*24*24

   
        conv2 = tf.layers.conv2d(r1,128,[5,5], strides=(1,1),padding = 'valid')
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain))#128*20*20

  
        conv3 = tf.layers.conv2d(r2,256,[5,5], strides=(1,1),padding = 'valid')
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain))#256*16*16

 
        conv4 = tf.layers.conv2d(r3,512,[4,4], strides=(2,2),padding = 'same')
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain))#512*8*8


        conv5 = tf.layers.conv2d(r4,1024,[4,4], strides=(2,2),padding = 'same')
        r5 = tf.nn.elu(tf.layers.batch_normalization(conv5,training=isTrain))#1024*4*4

       
        conv6 = tf.layers.conv2d(r5,1,[4,4], strides=(1,1),padding = 'valid')
        r6 = tf.nn.sigmoid(conv6)#1*1*1


        return r6



u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
ref = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='ref')
t = tf.placeholder(tf.float32, shape = (None, 28,28,1), name='t')
isTrain = tf.placeholder(dtype=tf.bool,name='isTrain') 
 

G_y = simple_G(u,ref,isTrain=isTrain,name='G_y')

    

D_real = simple_D(t,isTrain)
D_fake = simple_D(G_y,isTrain,reuse=True)

D_loss =  tf.reduce_mean(-0.5*(tf.log(D_real + 1e-8) + tf.log(1-D_fake + 1e-8)),name='D_loss')
gan_loss =  tf.reduce_mean(-0.5*(tf.log(D_fake + 1e-8)),name='gan_loss')
 
content_loss = tf.reduce_mean(0.5*(-t*tf.log(G_y + 1e-8) - (1-t)*tf.log(1-G_y + 1e-8)),name='content_loss')

G_loss = tf.add(content_loss, 0.001*gan_loss,name='G_loss')

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('D')]
G_vars = [var for var in T_vars if var.name.startswith('G')]

# When using the batchnormalization layers,
# it is necessary to manually add the update operations
# because the moving averages are not included in the graph
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :    
    D_optim = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list=D_vars, name='D_optim') 
    G_optim = tf.train.AdamOptimizer(0.00001).minimize(G_loss, var_list=G_vars, name='G_optim')





sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

np.random.seed(int(time.time()))

test_images,_ = mnist.test.next_batch(16)    
test_origin = test_images*np.random.uniform()

test_ref,_ = mnist.test.next_batch(16)
test_input = np.minimum(test_origin + test_ref*np.random.uniform() + np.random.uniform(size = (16,784)), 1.0)

my_lib.mnist_4by4_save(np.reshape(test_input,(-1,784)),file_name + '/input_noise.png')
my_lib.mnist_4by4_save(np.reshape(test_origin,(-1,784)),file_name + '/ground_true.png') 
my_lib.mnist_4by4_save(np.reshape(test_ref,(-1,784)),file_name + '/input_ref.png')
 

for i in range(1000000) :
    
    
    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform()
    train_ref,_ = mnist.train.next_batch(100)
    train_input = np.minimum(train_origin + train_ref*np.random.uniform() + np.random.uniform(size = (100,784)), 1.0)
    
    _ , D_e = sess.run([D_optim, D_loss],feed_dict={u : np.reshape(train_input,(-1,28,28,1)),
        ref : np.reshape(train_ref,(-1,28,28,1)), t : np.reshape(train_origin,(-1,28,28,1)),
        isTrain : True})
     
    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform()
    train_ref,_ = mnist.train.next_batch(100)
    train_input = np.minimum(train_origin + train_ref*np.random.uniform() + np.random.uniform(size = (100,784)), 1.0)
    
    _ , G_e, gan_e, content_e= sess.run([G_optim, G_loss, gan_loss, content_loss],
        feed_dict={u : np.reshape(train_input,(-1,28,28,1)), ref : np.reshape(train_ref,(-1,28,28,1)),
        t : np.reshape(train_origin,(-1,28,28,1)),isTrain : True})
 
   
    if i%1000 == 0:
        print('D_e : %.8f, G_e : %.8f, gan_e : %.8f, content_e : %.8f'%(D_e, G_e, gan_e, content_e))
        
        r = sess.run([G_y],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            ref : np.reshape(test_ref,(-1,28,28,1)), t : np.reshape(test_origin,(-1,28,28,1)),
            isTrain : False})
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))
      
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)









