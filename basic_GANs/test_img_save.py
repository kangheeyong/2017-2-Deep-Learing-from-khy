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
        plt.axis('off')         # 축 표시 제거
        ax.set_xticklabels([]) # x축 여백 제거
        ax.set_yticklabels([]) # y축 여백 제거
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig





resut_size = 16


if not os.path.exists('out/'):
    os.makedirs('out/')


for i in range(2) :
    imgs,_ = mnist.test.next_batch(16)

    fig = plot(imgs)
    #plt.show()
    plt.savefig('out/{}.png'.format(str(i).zfill(2)), bbox_inches='tight')
    plt.close(fig)

