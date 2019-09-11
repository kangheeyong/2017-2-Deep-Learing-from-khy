import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

corruption_level = 0.3


def corruption(dddd) : 
    mask = np.random.binomial(1,1- corruption_level,dddd.shape)
    corrupted_input = dddd*mask
    
    aaa = dddd - 1
    aaa = mask*aaa

    return corrupted_input + aaa


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
max_epoch = 100000
d_steps = 10
g_steps = 10
learning_rate = 0.0001
ramda = 100;

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 784])


D_W1 = tf.Variable(tf.random_normal(shape = [784, 512], mean = 0,stddev = 0.1))
D_b1 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.001))
D_W2 = tf.Variable(tf.random_normal(shape = [512, 256], mean = 0,stddev = 0.1))
D_b2 = tf.Variable(tf.random_normal(shape = [256], mean = 0,stddev = 0.001))
D_W3 = tf.Variable(tf.random_normal(shape = [256, 1], mean = 0,stddev = 0.1))
D_b3 = tf.Variable(tf.random_normal(shape = [1], mean = 0,stddev = 0.001))


G_W1 = tf.Variable(tf.random_normal(shape = [784, 512], mean = 0,stddev = 0.1))
G_b1 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.001))
G_W2 = tf.Variable(tf.random_normal(shape = [512, 256], mean = 0,stddev = 0.1))
G_b2 = tf.Variable(tf.random_normal(shape = [256], mean = 0,stddev = 0.001))
G_W3 = tf.Variable(tf.random_normal(shape = [256, 512], mean = 0,stddev = 0.1))
G_b3 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.001))
G_W4 = tf.Variable(tf.random_normal(shape = [512, 784], mean = 0,stddev = 0.1))
G_b4 = tf.Variable(tf.random_normal(shape = [784], mean = 0,stddev = 0.001))



theta_G = [G_W1, G_W2, G_W3,G_W4, G_b1, G_b2, G_b3,G_b4]
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]



def generator(z):
    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.tanh(tf.matmul(G_h2, G_W3) + G_b3)
    out  = tf.nn.sigmoid(tf.matmul(G_h3, G_W4) + G_b4)
    return out


def discriminator(x):
    D_h1 = tf.nn.tanh(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.tanh(tf.matmul(D_h1, D_W2) + D_b2)
    out = tf.matmul(D_h2, D_W3) + D_b3
    return out


if not os.path.exists('out_lsgans_L1_denoise1/'):
    os.makedirs('out_lsgans_L1_denoise1/')




G_sample = generator(Z)

D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = 0.5*tf.reduce_mean((D_real-1)**2 + D_fake**2)
G_loss = 0.5*tf.reduce_mean((D_fake-1)**2) + ramda*tf.reduce_mean(tf.abs(X - G_sample))

D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())



i = 0
test, _ = mnist.test.next_batch(resut_size)
test_corruption = corruption(test)        
for it in range(max_epoch) :
   

    for _ in range(d_steps):
        x_mb, _ = mnist.train.next_batch(mini_batch_size)
        z_mb = corruption(x_mb)

        _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: x_mb, Z: z_mb})

    for _ in range(g_steps) :


        _, G_loss_curr = sess.run([G_solver, G_loss],feed_dict={X : x_mb, Z: z_mb})


    if it % 500 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        if i == 0 :
            fig = plot(test_corruption)
            plt.savefig('out_lsgans_L1_denoise1/{}_corruption.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        samples = sess.run(G_sample, feed_dict={Z: test})

        fig = plot(samples)
        plt.savefig('out_lsgans_L1_denoise1/{}_gener.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)





