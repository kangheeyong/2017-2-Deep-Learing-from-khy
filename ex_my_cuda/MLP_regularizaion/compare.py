import tensorflow as tf
import numpy as np
import os
import time


start = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)

mini_batch_size = 128
max_epoch = 20000
learning_rate = 0.2


X = tf.placeholder(tf.float32, shape=[None, 784])
T = tf.placeholder(tf.float32, shape=[None, 10])



W1 = tf.Variable(tf.random_normal(shape = [784, 512], mean = 0,stddev = 0.1))
b1 = tf.Variable(tf.random_normal(shape = [512], mean = 0,stddev = 0.001))
W2 = tf.Variable(tf.random_normal(shape = [512, 128], mean = 0,stddev = 0.1))
b2 = tf.Variable(tf.random_normal(shape = [128], mean = 0,stddev = 0.001))
W3 = tf.Variable(tf.random_normal(shape = [128, 32], mean = 0,stddev = 0.1))
b3 = tf.Variable(tf.random_normal(shape = [32], mean = 0,stddev = 0.001))
W4 = tf.Variable(tf.random_normal(shape = [32, 10], mean = 0,stddev = 0.1))
b4 = tf.Variable(tf.random_normal(shape = [10], mean = 0,stddev = 0.001))




a2 = tf.nn.relu(tf.matmul(X, W1) + b1)
a3 = tf.nn.relu(tf.matmul(a2, W2) + b2)
a4 = tf.nn.relu(tf.matmul(a3, W3) + b3)
a5 = tf.nn.sigmoid(tf.matmul(a4, W4) + b4)
Y = a5 




loss = -tf.reduce_mean(T*tf.log(Y + 0.000000001) + (1-T)*tf.log(1-Y + 0.000000001) )

solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.arg_max(Y,1),tf.arg_max(T,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())


validation_input, validation_label = mnist.test.next_batch(5000);
test_input, test_label = mnist.test.next_batch(5000);

for it in range(max_epoch) :

    train_input, train_label = mnist.train.next_batch(mini_batch_size)

    _, train_error,train_acc = sess.run([solver, loss,accuracy],feed_dict={X : train_input, T: train_label})

    if(it%1000 == 0) :
        validation_error,validation_acc = sess.run([loss,accuracy],feed_dict = {X : validation_input , T : validation_label})
        print("tr e: ",train_error,", tr acc: ",train_acc,", val e: ",validation_error,", val acc: ",validation_acc)        

test_error,test_acc = sess.run([loss,accuracy],feed_dict = {X : test_input , T : test_label})
      
print("test e: ",test_error,", test acc: ",test_acc)


end = time.time()-start

print("total time : ",end)



