import tensorflow as tf
import numpy as np
import os
import sys
import time
import my_lib 
import time 

os.environ["CUDA_VISIBLE_DEVICES"]="0"



start = time.time()

  
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('_load.py')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)

    
sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(file_name + '/para.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(file_name + '/'))
    
 
one_hot = np.eye(10)


z = sess.graph.get_tensor_by_name("z:0")
u = sess.graph.get_tensor_by_name("u:0")
z_c = sess.graph.get_tensor_by_name("z_c:0")
z_fill = sess.graph.get_tensor_by_name("z_fill:0")
isTrain = sess.graph.get_tensor_by_name("isTrain:0")

    
G_z = sess.graph.get_tensor_by_name("G_z:0")

D_loss = sess.graph.get_tensor_by_name("D_loss:0")
G_loss = sess.graph.get_tensor_by_name("G_loss:0")

D_optim = sess.graph.get_operation_by_name("D_optim")
G_optim = sess.graph.get_operation_by_name("G_optim")


np.random.seed(int(time.time()))
for i in range(10) :

 
    test_z = np.random.normal(0,1,size=(16,1,1,100))
    test_temp = np.random.randint(0,9,(16,1))
    test_z_c = one_hot[test_temp].reshape([-1,1,1,10])
    np.savetxt(file_name+'/gen_result_{}condition.txt'.format(str(i).zfill(3)),test_temp,fmt='%d')


    r = sess.run([G_z],feed_dict={z : test_z,z_c : test_z_c, isTrain : False})        
    my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),
            file_name + '/gen_result_{}.png'.format(str(i).zfill(3)))





end = time.time()-start

print("total time : ",end)
















