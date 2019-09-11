import tensorflow as tf
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="1"


def simple_nn(x) :
    # initializers
 
    with tf.variable_scope('nn') :
            
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev = 0.02)
        b_init = tf.constant_initializer(0.0)

    # 1st layer
        w1 = tf.get_variable('w1',[x.get_shape()[1],6],initializer = w_init)
        b1 = tf.get_variable('b1',[6],initializer = b_init)
        h1 = tf.nn.elu(tf.matmul(x,w1)+b1)


    # 2st layer
        w2 = tf.get_variable('w2',[h1.get_shape()[1],4],initializer = w_init)
        b2 = tf.get_variable('b2',[4],initializer = b_init)
        h2 = tf.nn.elu(tf.matmul(h1,w2)+b2)


    # 3st layer
        w3 = tf.get_variable('w3',[h2.get_shape()[1],1],initializer = w_init)
        b3 = tf.get_variable('b3',[1],initializer = b_init)
        h3 = tf.nn.sigmoid(tf.matmul(h2,w3)+b3)

    return h3



xor_input = np.array([[0,0],[0,1], [1,0], [1,1]])
xor_output = np.array([[1],[0],[0],[1]])


with tf.device('/gpu:0') :
    u = tf.placeholder(tf.float32, shape = (None, 2),name='u')
    t = tf.placeholder(tf.float32, shape = (None, 1), name='t')
    y = simple_nn(u)

    loss = tf.reduce_mean(0.5*(-t*tf.log(y + 1e-8) - (1-t)*tf.log(1-y + 1e-8)),name='loss')
    optim = tf.train.AdamOptimizer(0.001).minimize(loss,name='optim')




sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())



for i in range(2000) :
            
    _ , e = sess.run([ optim, loss],feed_dict={u : xor_input, t :xor_output})

    print(e)




if not os.path.isdir('ex_basic_1') :
    os.mkdir('ex_basic_1')

saver = tf.train.Saver()
saver.save(sess,'ex_basic_1/example_1.cktp')








