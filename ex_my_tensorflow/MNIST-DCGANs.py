import tensorflow as tf
import numpy as np
import time 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True,reshape=[])

def lrelu(x,th=0.2) :
    return tf.maximum(th*x,x)

def simple_G(x, isTrain = True, reuse = False) : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('G',reuse=reuse) :
        
        #x = (-1, 1, 1, 100)
        conv1 = tf.layers.conv2d_transpose(x,1024,[4,4], strides=(1,1),padding = 'valid')
        r1 = lrelu(tf.layers.batch_normalization(conv1,training=isTrain),0.2)#1024*4*4
        
        conv2 = tf.layers.conv2d_transpose(r1,512,[4,4], strides=(2,2),padding = 'same')
        r2 = lrelu(tf.layers.batch_normalization(conv2,training=isTrain),0.2)#512*8*8
        
        conv3 = tf.layers.conv2d_transpose(r2,256,[4,4], strides=(2,2),padding = 'same')
        r3 = lrelu(tf.layers.batch_normalization(conv3,training=isTrain),0.2)#256*16*16
        
        conv4 = tf.layers.conv2d_transpose(r3,128,[4,4], strides=(2,2),padding = 'same')
        r4 = lrelu(tf.layers.batch_normalization(conv4,training=isTrain),0.2)#128*32*32

        conv5 = tf.layers.conv2d_transpose(r4,1,[4,4], strides=(2,2),padding = 'same')
        r5 = tf.nn.tanh(conv5)#1*64*64
        
        return r5

def simple_D(x,isTrain=True, reuse = False) :
    
    with tf.variable_scope('D', reuse=reuse) :
        
        #x = (-1,28,28,1)
        conv1 = tf.layers.conv2d(x,128,[4,4], strides=(2,2),padding = 'same')
        r1 = lrelu(conv1)#128*32*32
        
        conv2 = tf.layers.conv2d(r1,256,[4,4], strides=(2,2),padding = 'same')
        r2 = lrelu(tf.layers.batch_normalization(conv2,training=isTrain),0.2)#256*16*16
        
        conv3 = tf.layers.conv2d(r2,512,[4,4], strides=(2,2),padding = 'same')
        r3 = lrelu(tf.layers.batch_normalization(conv3,training=isTrain),0.2)#128*8*8
        
        conv4 = tf.layers.conv2d(r3,1024,[4,4], strides=(2,2),padding = 'same')
        r4 = lrelu(tf.layers.batch_normalization(conv4,training=isTrain),0.2)#1024*4*4

        conv5 = tf.layers.conv2d(r4,1,[4,4], strides=(1,1),padding = 'valid')
        r5 = tf.nn.sigmoid(conv5)#1*1*1
        


        return r5, conv5

batch_size = 100
lr = 0.0002
train_epoch = 100



z = tf.placeholder(tf.float32,shape=(None,1,1,100))    
x = tf.placeholder(tf.float32, shape = (None, 64,64,1))
isTrain = tf.placeholder(dtype=tf.bool)
    
G_z = simple_G(z,isTrain)

D_real, D_real_logits = simple_D(x,isTrain)
D_fake, D_fake_logits = simple_D(G_z,isTrain,reuse=True)


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logits,
    labels=tf.ones([batch_size,1,1,1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits,
    labels=tf.zeros([batch_size,1,1,1])))
D_loss = D_loss_real + D_loss_fake
    
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits,
    labels=tf.ones([batch_size,1,1,1])))




T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('D')]
G_vars = [var for var in T_vars if var.name.startswith('G')]


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :    
    D_optim = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(G_loss, var_list=G_vars)
    
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
tf.global_variables_initializer().run()


train_set = tf.image.resize_images(mnist.train.images,[64,64]).eval()
train_set = (train_set -0.5)/0.5





np.random.seed(int(time.time()))


for epoch in range(train_epoch) :
    G_losses = []
    D_losses = []
    for iter in range(mnist.train.num_examples // batch_size) :

        x_ = train_set[iter*batch_size : (iter+1)*batch_size]
        z_ = np.random.normal(0,1, (batch_size,1,1,100))
        
        loss_d_,_ = sess.run([D_loss,D_optim],{x:x_,z:z_,isTrain : True})
        D_losses.append(loss_d_)

        z_ = np.random.normal(0,1, (batch_size,1,1,100))
        
        loss_g_,_ = sess.run([G_loss,G_optim],{x:x_,z:z_, isTrain : True})
        G_losses.append(loss_g_)

    print('d_loss : %.8f, g_loss : %.8f'%(np.mean(D_losses), np.mean(G_losses)))






























