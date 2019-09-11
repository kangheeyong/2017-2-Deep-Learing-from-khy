import tensorflow as tf
import numpy as np
import os
import time


start = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)

mini_batch_size = 300
max_epoch = 1000000
learning_rate = 0.00009


X = tf.placeholder(tf.float32, shape=[None, 784])
T = tf.placeholder(tf.float32, shape=[None, 10])
input_drop = tf.placeholder(tf.float32)
hidden_drop = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_normal(shape = [784, 1024], mean = 0,stddev = 0.1))
b1 = tf.Variable(tf.random_normal(shape = [1024], mean = 0,stddev = 0.1))

W2 = tf.Variable(tf.random_normal(shape = [1024, 1024], mean = 0,stddev = 0.1))
b2 = tf.Variable(tf.random_normal(shape = [1024], mean = 0,stddev = 0.1))

W3 = tf.Variable(tf.random_normal(shape = [1024, 2048], mean = 0,stddev = 0.1))
b3 = tf.Variable(tf.random_normal(shape = [2048], mean = 0,stddev = 0.1))

W4 = tf.Variable(tf.random_normal(shape = [2048, 10], mean = 0,stddev = 0.1))
b4 = tf.Variable(tf.random_normal(shape = [10], mean = 0,stddev = 0.1))



X_drop = tf.nn.dropout(X,input_drop)
a2 = tf.nn.relu(tf.matmul(X_drop, W1) + b1)

a2_drop = tf.nn.dropout(a2,hidden_drop)
a3 = tf.nn.relu(tf.matmul(a2_drop, W2) + b2)

a3_drop = tf.nn.dropout(a3,hidden_drop)
a4 = tf.nn.relu(tf.matmul(a3_drop, W3) + b3)

a5 = tf.nn.sigmoid(tf.matmul(a4, W4) + b4)
Y = a5




loss = -tf.reduce_mean(T*tf.log(Y + 0.000000001) + (1-T)*tf.log(1-Y + 0.000000001) )

solver = tf.train.AdamOptimizer(learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.arg_max(Y,1),tf.arg_max(T,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())


validation_input, validation_label = mnist.test.next_batch(5000);
test_input, test_label = mnist.test.next_batch(5000);

for it in range(max_epoch) :

    train_input, train_label = mnist.train.next_batch(mini_batch_size)

    _, train_error,train_acc = sess.run([solver, loss,accuracy],feed_dict={X : train_input, T: train_label,
        input_drop : 1.0, hidden_drop : 1.0})

    if(it%1000 == 0) :
        validation_error,validation_acc = sess.run([loss,accuracy],feed_dict = {X : validation_input , 
            T : validation_label,input_drop : 1.0, hidden_drop : 1.0})
        print("tr e: ",train_error,", tr acc: ",train_acc,", val e: ",validation_error,", val acc: ",validation_acc)        

test_error,test_acc = sess.run([loss,accuracy],feed_dict = {X : test_input , T : test_label,
    input_drop : 1.0, hidden_drop : 1.0})
      
print("test e: ",test_error,", test acc: ",test_acc)


end = time.time()-start

print("total time : ",end)



