import tensorflow as tf
import numpy as np
import os

   
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)



sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('ex_basic_2/example_2.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('ex_basic_2/'))
    
 
u = sess.graph.get_tensor_by_name("u:0")
t = sess.graph.get_tensor_by_name("t:0")
loss = sess.graph.get_tensor_by_name("loss:0")
optim = sess.graph.get_operation_by_name("optim")
accuracy = sess.graph.get_tensor_by_name("accuracy:0")




for i in range(20) :
    
    train_input, train_label = mnist.train.next_batch(100)
    a = np.reshape(train_input,(-1,28,28,1))
            
    _ , e, acc = sess.run([ optim, loss, accuracy],feed_dict={u : a, t : train_label})

    print('e : %.6f, acc : %.6f'%(e,acc))




#if not os.path.isdir('ex_basic_2') :
#    os.mkdir('ex_basic_2')

#saver = tf.train.Saver()
#saver.save(sess,'ex_basic_2/example_2.cktp')



