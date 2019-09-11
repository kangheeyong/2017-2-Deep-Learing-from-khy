import tensorflow as tf
import numpy as np
import os
import sys
import time
import my_lib 
import time 

os.environ["CUDA_VISIBLE_DEVICES"]="1"



start = time.time()

  
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('.')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

def simple_G(x,c,isTrain = True, reuse = False, name = 'G_out') : #input = (minibatch * w * h * ch)
    
    # out size = (in size + 2*padding - kenel)/strides + 1    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)


    with tf.variable_scope('G',reuse=reuse) :
        
        #x = (-1, 1, 1, 100)
        #c = (-1, 1, 1, 10)

        cat1 = tf.concat([x,c],3)

        conv1 = tf.layers.conv2d_transpose(cat1,1024,[4,4], strides=(1,1),padding = 'valid',
                kernel_initializer=w_init, bias_initializer=b_init) 
        r1 = tf.nn.elu(tf.layers.batch_normalization(conv1,training=isTrain))#1024*4*4
        
        conv2 = tf.layers.conv2d_transpose(r1,512,[4,4], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain))#512*8*8
                
        conv3 = tf.layers.conv2d_transpose(r2,256,[4,4], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain))#256*16*16

        conv4 = tf.layers.conv2d_transpose(r3,128,[4,4], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain))#128*32*32

        conv5 = tf.layers.conv2d(r4,64,[3,3], strides=(1,1),padding = 'valid',
                kernel_initializer=w_init, bias_initializer=b_init)
        r5 = tf.nn.elu(tf.layers.batch_normalization(conv5,training=isTrain))#64*30*30

        conv6 = tf.layers.conv2d(r5,1,[3,3], strides=(1,1),padding = 'valid',
                kernel_initializer=w_init, bias_initializer=b_init)
    r6 = tf.nn.sigmoid(conv6,name=name)#1*28*28
  

    return r6

def simple_D(x,c_fill,isTrain=True,reuse = False) :
    
    with tf.variable_scope('D', reuse=reuse) :
        
        #x = (-1,28,28,1)
        #c_fill = (-1, 28, 28,10)

        cat1 = tf.concat([x,c_fill],3)


        conv1 = tf.layers.conv2d(cat1,64,[5,5], strides=(1,1),padding = 'valid')
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


def simple_cnn(x) :
    
    w_init = tf.truncated_normal_initializer(mean=0.0, stddev = 0.02)
    b_init = tf.constant_initializer(0.0)

    # out size = (in size + 2*padding - kenel)/strides + 1    

    with tf.variable_scope('cnn') :
        
        
        conv1 = tf.layers.conv2d(x,32,[5,5], strides=(1,1),padding = 'valid')
        r1 = tf.nn.elu(conv1)#32*24*24
        
        conv2 = tf.layers.conv2d(r1,64,[5,5], strides=(1,1),padding = 'valid')
        r2 = tf.nn.elu(conv2)#64*20*20
        
        conv3 = tf.layers.conv2d(r2,128,[5,5], strides=(1,1),padding = 'valid')
        r3 = tf.nn.elu(conv3)#128*16*16
        
        conv4 = tf.layers.conv2d(r3,256,[4,4], strides=(2,2),padding = 'same')
        r4 = tf.nn.elu(conv4)#256*8*8

        conv5 = tf.layers.conv2d(r4,512,[4,4], strides=(2,2),padding = 'same')
        r5 = tf.nn.elu(conv5)#512*4*4
        
        conv6 = tf.layers.conv2d(r5,1024,[4,4], strides=(1,1),padding = 'valid')
        r6 = tf.nn.elu(conv6)#1024*1*1

        fc = tf.reshape(r6,[-1,1024*1*1])       

        # 1st layer
        w1 = tf.get_variable('w1',[fc.get_shape()[1],1024],initializer = w_init)
        b1 = tf.get_variable('b1',[1024],initializer = b_init)
        fc1 = tf.nn.elu(tf.matmul(fc,w1)+b1)

         # 1st layer
        w2 = tf.get_variable('w2',[fc1.get_shape()[1],10],initializer = w_init)
        b2 = tf.get_variable('b2',[10],initializer = b_init)
        fc2 = tf.nn.sigmoid(tf.matmul(fc1,w2)+b2)

        return fc2


cnn_u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='cnn_u')
cnn_t = tf.placeholder(tf.float32, shape = (None, 10), name='cnn_t')
cnn_y = simple_cnn(cnn_u)

cnn_loss = tf.reduce_mean(0.5*(-cnn_t*tf.log(cnn_y + 1e-8) - (1-cnn_t)*tf.log(1-cnn_y + 1e-8)),name='cnn_loss')
cnn_optim = tf.train.AdamOptimizer(0.0001).minimize(cnn_loss,name='cnn_optim')
    
cnn_correct_prediction = tf.equal(tf.argmax(cnn_y,1),tf.argmax(cnn_t,1))
cnn_accuracy = tf.reduce_mean(tf.cast(cnn_correct_prediction, tf.float32),name = 'cnn_accuracy')







z = tf.placeholder(tf.float32,shape=(None,1,1,100),name = 'z')    
u = tf.placeholder(tf.float32, shape = (None, 28,28,1),name='u')
z_c =  tf.placeholder(tf.float32,shape=(None,1,1,10),name = 'z_c')    
z_fill = tf.placeholder(tf.float32,shape=(None,28,28,10),name = 'z_fill')    
one_hot = np.eye(10)


isTrain = tf.placeholder(dtype=tf.bool,name='isTrain') 
    
G_z = simple_G(z,z_c,name='G_z')

D_real = simple_D(u,z_fill,isTrain)
D_fake = simple_D(G_z,z_fill,isTrain,reuse=True)


D_loss =  tf.reduce_mean(-0.5*(tf.log(D_real + 1e-8) + tf.log(1-D_fake + 1e-8)),name='D_loss')
G_loss =  tf.reduce_mean(-0.5*(tf.log(D_fake + 1e-8)),name='G_loss')
 

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

test_z = np.random.normal(0,1,size=(16,1,1,100))
test_temp = np.random.randint(0,9,(16,1))
test_z_c = one_hot[test_temp].reshape([-1,1,1,10])
np.savetxt(file_name+'/condition.txt',test_temp,fmt='%d')

test_input = np.reshape(mnist.test.images,(-1, 28, 28, 1))
test_label = mnist.test.labels

hist_G = []
hist_D = []
for i in range(100000) :
    G_errer = []
    D_errer = []

    train_images,train_labels = mnist.train.next_batch(100)    
    z_c_ = np.reshape(train_labels,(-1,1,1,10))    
    z_fill_ = z_c_*np.ones([100,28,28,10])
    u_ = np.reshape(train_images,(-1,28,28,1)) 
    z_ = np.random.normal(0,1,size=(100,1,1,100))

    _ , D_e = sess.run([D_optim, D_loss], {u : u_,z_fill : z_fill_, z : z_,z_c : z_c_, isTrain : True})
    D_errer.append(D_e)
    
    while True :
        train_images,train_labels = mnist.train.next_batch(100)    
        z_c_ = np.reshape(train_labels,(-1,1,1,10))    
        z_fill_ = z_c_*np.ones([100,28,28,10])
        u_ = np.reshape(train_images,(-1,28,28,1)) 
        z_ = np.random.normal(0,1,size=(100,1,1,100))

        _ , G_e = sess.run([G_optim, G_loss], {u : u_,z_fill : z_fill_, z : z_,z_c : z_c_, isTrain : True})
        G_errer.append(G_e)
        if G_e < 2*np.log(2) :
            break

    cnn_img, cnn_label = mnist.train.next_batch(100)
    cnn_img = np.reshape(cnn_img,(-1,28,28,1))
    cnn_label = np.reshape(cnn_label,(-1,10))

    cnn_z = np.random.normal(0,1,size=(100,1,1,100))
    cnn_temp = np.random.randint(0,9,(100,1))
    cnn_z_c = one_hot[cnn_temp].reshape([-1,1,1,10])
     
    gen_d = sess.run([G_z],feed_dict={z : cnn_z,z_c : cnn_z_c, isTrain : False})        
     
    cnn_input = np.concatenate([cnn_img,np.reshape(gen_d,(-1,28,28,1))],0)
    cnn_target = np.concatenate([cnn_label,np.reshape(cnn_z_c,(-1,10))],0)
    

    _ , e, acc = sess.run([ cnn_optim, cnn_loss, cnn_accuracy],
            feed_dict={cnn_u : cnn_input, cnn_t : cnn_target})
   
    if i%1000 == 0:

        hist_D.append(np.mean(D_errer))
        hist_G.append(np.mean(G_errer))
        test_acc = sess.run([cnn_accuracy],feed_dict={cnn_u : test_input, cnn_t : test_label})
     
        print('D_e : %.2f, G_e : %.2f, cnn_t_e : %.2f, cnn_t_acc : %.2f,test acc : '
                %(np.mean(D_errer),np.mean(G_errer),e,acc),test_acc)
                
        r = sess.run([G_z],feed_dict={z : test_z,z_c : test_z_c, isTrain : False})        
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))



my_lib.gan_loss_graph_save(G_loss = hist_G,D_loss=hist_D,path = file_name + '/loss_graph.png')   
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)
















