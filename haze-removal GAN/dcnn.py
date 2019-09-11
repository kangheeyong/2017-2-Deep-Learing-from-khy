import tensorflow as tf
import numpy as np
import os
import sys
import time
import time 
import pickle
import gzip

#os.environ["CUDA_VISIBLE_DEVICES"]="1"


start = time.time()

with gzip.open('hazed_images.pickle.gzip','rb') as f :
    hazed_images = pickle.load(f)


with gzip.open('origin_images.pickle.gzip','rb') as f :
    origin_images = pickle.load(f)

print(origin_images.shape)
print(hazed_images.shape)

 


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

def simple_G(x,isTrain = True, reuse = False, name = 'G_out') : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('G',reuse=reuse) :
        
        #x = (-1, 256, 256, 3)
        conv1 = tf.layers.conv2d(x,64,[3,3], strides=(1,1),padding = 'same') 
        r1 = tf.nn.elu(tf.layers.batch_normalization(conv1,training=isTrain))#64*256*256
        
        conv2 = tf.layers.conv2d(r1,64,[3,3], strides=(1,1),padding = 'same')
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain))#64*256*256
        
        conv3 = tf.layers.conv2d(r2,64,[3,3], strides=(1,1),padding = 'same')
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain))#64*256*256

        conv4 = tf.layers.conv2d(r3,64,[3,3], strides=(1,1),padding = 'same')
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain))#64*256*256

        conv5 = tf.layers.conv2d(r4,64,[3,3], strides=(1,1),padding = 'same')
        r5 = tf.nn.elu(tf.layers.batch_normalization(conv5,training=isTrain))#64*256*256
        
        conv6 = tf.layers.conv2d(r5,64,[3,3], strides=(1,1),padding = 'same')
        r6 = tf.nn.elu(tf.layers.batch_normalization(conv6,training=isTrain))#64*256*256

        conv7 = tf.layers.conv2d(r6,64,[3,3], strides=(1,1),padding = 'same')
        r7 = tf.nn.elu(tf.layers.batch_normalization(conv7,training=isTrain))#64*256*256


        conv8 = tf.layers.conv2d(r7,3,[3,3], strides=(1,1),padding = 'same')
    r8 = tf.nn.elu(conv8,name=name)#3*256*256
  

    return r8



u = tf.placeholder(tf.float32, shape = (None, 256,256,3),name='u')
t =  tf.placeholder(tf.float32, shape = (None, 256,256,3),name='t')
isTrain = tf.placeholder(dtype=tf.bool,name='isTrain') 
    
y = simple_G(u,isTrain = isTrain, name='y')


G_loss =  tf.reduce_mean(0.5*tf.square(t - y),name='G_loss')
 
 

T_vars = tf.trainable_variables()
G_vars = [var for var in T_vars if var.name.startswith('G')]

    # When using the batchnormalization layers,
    # it is necessary to manually add the update operations
    # because the moving averages are not included in the graph
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :    
    G_optim = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list=G_vars, name='G_optim')




sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

minibatch = 15


hist_G = []
for epoch in range(1000) :
    
    it = np.random.randint(0,471 - minibatch)
        
    in_images = origin_images[it : it + minibatch]
    out_images = hazed_images[it : it + minibatch]



    _ , G_e = sess.run([G_optim, G_loss],{u : in_images, t : out_images, isTrain : True})

    hist_G.append(G_e)


    print('G_e : %.8f'%(G_e))


saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)



