import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
   
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


resut_size = 16
mini_batch_size = 128
max_epoch = 1000
learning_rate = 0.0001


X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 784])



G_W1 = tf.Variable(tf.random_normal(shape = [784, 512], mean = 0,stddev = 0.1))
G_b1 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.001))
G_W2 = tf.Variable(tf.random_normal(shape = [512, 256], mean = 0,stddev = 0.1))
G_b2 = tf.Variable(tf.random_normal(shape = [256], mean = 0,stddev = 0.001))
G_W3 = tf.Variable(tf.random_normal(shape = [256, 512], mean = 0,stddev = 0.1))
G_b3 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.001))
G_W4 = tf.Variable(tf.random_normal(shape = [512, 784], mean = 0,stddev = 0.1))
G_b4 = tf.Variable(tf.random_normal(shape = [784], mean = 0,stddev = 0.001))



theta_G = [G_W1, G_W2, G_W3,G_W4, G_b1, G_b2, G_b3,G_b4]



def generator(z):
    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.tanh(tf.matmul(G_h2, G_W3) + G_b3)
    out  = tf.nn.sigmoid(tf.matmul(G_h3, G_W4) + G_b4)
    return out



if not os.path.exists('out_autoencoder/'):
    os.makedirs('out_autoencoder/')




G_sample = generator(Z)


G_loss = -tf.reduce_mean(X*tf.log(G_sample) + (1-X)*tf.log(1-G_sample + 0.000000001) )

G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())



i = 0
test, _ = mnist.test.next_batch(resut_size)
        
for it in range(max_epoch) :
   

    x_mb, _ = mnist.train.next_batch(mini_batch_size)
    z_mb = x_mb



    _, G_loss_curr = sess.run([G_solver, G_loss],feed_dict={X : z_mb, Z: z_mb})


    if it % 500 == 0:
        print('Iter: {}; G_loss: {:.4}'.format(it, G_loss_curr))

        if i == 0 :
            fig = plot(test)
            plt.savefig('out_autoencoder/{}_origin.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        samples = sess.run(G_sample, feed_dict={Z: test})

        fig = plot(samples)
        plt.savefig('out_autoencoder/{}_gener.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)





