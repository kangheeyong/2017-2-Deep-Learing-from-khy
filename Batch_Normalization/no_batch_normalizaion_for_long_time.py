import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
if len(sys.argv) != 2:
    print("USAGE : ",sys.argv[0]," [seed number]")
    sys.exit()


my_seed = 1



folder_name = sys.argv[1]


np.random.seed(my_seed)


if not os.path.exists('no_BN_' + folder_name):
    os.makedirs('no_BN_' + folder_name)


mini_batch_size = 64
max_iteration = 1000000
learning_rate = int(sys.argv[1])/100
print(learning_rate)

epsilon = 1e-8



X = tf.placeholder(tf.float32, shape=[None, 784])
T = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.random_normal(shape = [784, 512], mean = 0,stddev = 0.1,seed = my_seed))
b1 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.1,seed = 2*my_seed))
W2 = tf.Variable(tf.random_normal(shape = [512, 256], mean = 0,stddev = 0.1,seed = 3*my_seed))
b2 = tf.Variable(tf.random_normal(shape = [256], mean = 0,stddev = 0.1,seed = 4*my_seed))
W3 = tf.Variable(tf.random_normal(shape = [256, 10], mean = 0,stddev = 0.1,seed = 5*my_seed))
b3 = tf.Variable(tf.random_normal(shape = [10], mean = 0,stddev = 0.1, seed = 6*my_seed))


z2 = tf.matmul(X,W1) + b1
a2 = tf.nn.relu(z2)


z3 = tf.matmul(a2,W2) + b2
a3 = tf.nn.relu(z3)


z4 = tf.matmul(a3,W3) + b3
a4 = tf.nn.sigmoid(z4)

y = a4


cross_entropy = -tf.reduce_mean(T*tf.log(y + epsilon) +(1-T)*tf.log(1-y + epsilon))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(T,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())


mean1, mean2, mean3 = [], [], []
train_error, train_accuracy = [], []
validation_error, validation_accuracy = [],[]

test_input, test_label = mnist.test.next_batch(1000)

for i in range(max_iteration) :
    
    train_input, train_label = mnist.train.next_batch(mini_batch_size)

    res = sess.run([train_step,cross_entropy,accuracy],feed_dict = {X : train_input , T : train_label})
    res2 = sess.run([cross_entropy,accuracy,z2,z3,z4],feed_dict = {X : test_input, T : test_label})
    
    if i % 1000 == 0 :

        train_error.append(res[1])
        train_accuracy.append(res[2])
        validation_error.append(res2[0])
        validation_accuracy.append(res2[1])
        print(i, res[1], res[2], res2[0],res2[1])



np.savetxt(fname='no_BN_' + folder_name + '/train_error.txt',X=train_error,fmt="%f")
np.savetxt(fname='no_BN_' + folder_name + '/train_accuracy.txt',X=train_accuracy,fmt="%f")
np.savetxt(fname='no_BN_' + folder_name + '/validation_error.txt',X=validation_error,fmt="%f")
np.savetxt(fname='no_BN_' + folder_name + '/validation_accuracy.txt',X=validation_accuracy,fmt="%f")
#np.savetxt(fname='no_BN_' + folder_name + '/mean1.txt',X=mean1,fmt="%f")
#np.savetxt(fname='no_BN_' + folder_name + '/mean2.txt',X=mean2,fmt="%f")
#np.savetxt(fname='no_BN_' + folder_name + '/mean3.txt',X=mean3,fmt="%f")

















