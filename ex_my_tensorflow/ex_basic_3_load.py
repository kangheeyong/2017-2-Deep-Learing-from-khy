import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split("_load.py")[0]
if not os.path.isdir(file_name) :
    os.mkdir(file_name)
    
sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(file_name + '/para.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(file_name + '/'))
    
 
u = sess.graph.get_tensor_by_name("u:0")
t = sess.graph.get_tensor_by_name("t:0")
y = sess.graph.get_tensor_by_name("y:0")
loss = sess.graph.get_tensor_by_name("loss:0")
optim = sess.graph.get_operation_by_name("optim")






test_origin,_ = mnist.test.next_batch(16)    
test_input = np.minimum(test_origin +  np.random.uniform(size = (16,784)), 1.0)
  
for i in range(2001) :
    
    train_origin,_ = mnist.train.next_batch(100)
    
    train_input = np.minimum(train_origin +  np.random.uniform(size = (100,784)), 1.0)
    
    _ , e = sess.run([ optim, loss],
            feed_dict={u : np.reshape(train_input,(-1,28,28,1)), t : np.reshape(train_origin,(-1,28,28,1))})
    
    if i%1000 == 0:
        print('e : %.8f'%(e))

        r = sess.run([y],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
            t : np.reshape(test_origin,(-1,28,28,1))})
   
           
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))
       
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')









