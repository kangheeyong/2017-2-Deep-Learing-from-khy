import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05) #이미지 사이간격 조절

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
 
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig



start = time.time()


   

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

with tf.device('/gpu:0') :
    u = tf.placeholder(tf.float32, shape = (None, 1568),name='u')
    t = tf.placeholder(tf.float32, shape = (None, 784), name='t')
    y = simple_resnn(u)

    loss = 0.5*tf.reduce_mean(-t*tf.log(y + 1e-8) - (1-t)*tf.log(1-y + 1e-8),name='loss')
    optim = tf.train.AdamOptimizer(0.0001).minimize(loss,name='optim')
    





sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

test_images,_ = mnist.test.next_batch(16)    
test_origin = test_images*np.random.uniform()

test_ref,_ = mnist.test.next_batch(16)
test_temp = np.minimum(test_origin + test_ref*np.random.uniform() + np.random.uniform(size = (16,784)), 1.0)
test_input = np.concatenate((test_temp, test_ref),axis=1)



if not os.path.isdir('resNet_v1_compare') :
    os.mkdir('resNet_v1_compare')


fig = plot(test_temp)
plt.savefig('resNet_v1_compare/input_noise.png', bbox_inches='tight')
plt.close(fig)

fig = plot(test_origin)
plt.savefig('resNet_v1_compare/ground_true.png', bbox_inches='tight')
plt.close(fig)

fig = plot(test_ref)
plt.savefig('resNet_v1_compare/input_ref.png', bbox_inches='tight')
plt.close(fig)


  
for i in range(100000) :
    
    train_images,_ = mnist.train.next_batch(100)
    train_origin = train_images * np.random.uniform()
    train_ref,_ = mnist.train.next_batch(100)
    
    temp = np.minimum(train_origin + train_ref*np.random.uniform() + np.random.uniform(size = (100,784)), 1.0)
    
    train_input = np.concatenate((temp, train_ref),axis=1)
    _ , e = sess.run([ optim, loss],feed_dict={u : train_input, t : train_origin})
    
    if i%10000 == 0:
        print('e : %.8f'%(e))

        r = sess.run([y],feed_dict={u : test_input, t : test_origin})
   
        fig = plot(np.reshape(r,(-1,784)))
        plt.savefig('resNet_v1_compare/result_{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig)
      
saver = tf.train.Saver()
saver.save(sess,'resNet_v1_compare/para.cktp')





end = time.time()-start

print("total time : ",end)


