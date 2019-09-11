import tensorflow as tf
import numpy as np
import os


print(tf.__version__)

xor_input = np.array([[0,0],[0,1], [1,0], [1,1]])
xor_output = np.array([[1],[0],[0],[1]])



sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph('ex_basic_1/example_1.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('ex_basic_1/'))
    
x = input() 
u = sess.graph.get_tensor_by_name("u:0")
t = sess.graph.get_tensor_by_name("t:0")
loss = sess.graph.get_tensor_by_name("loss:0")
optim = sess.graph.get_operation_by_name("optim")





for i in range(1000) :
            
    _, e = sess.run([optim, loss],feed_dict={u : xor_input, t :xor_output})
    print(e)

           



#if not os.path.isdir('ex_basic_1') :
#    os.mkdir('ex_basic_1')

#saver = tf.train.Saver()
#saver.save(sess,'ex_basic_1/example_1.cktp')






