#ifndef __MY_MNIST_CLASS_CPP__
#define __MY_MNIST_CLASS_CPP__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "my_graph_net_sub.cuh"

#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column
#define MAX_TRAIN_INX       55000
#define MAX_LABEL_NUM       10
#define MAX_IMAGE_SIZE      784



class EXAMPLE_MNIST_CLASS
{
    private : 


        FILE *fd_t10k_images;
        FILE *fd_t10k_labels;
        FILE *fd_train_images;
        FILE *fd_train_labels;

        int number_of_batch;
        int cur_index;        
        int cur_point;
        float *train_images;
        float *train_labels;

        unsigned char *buff;
    public :     
        float *validation_images;
        float *validation_labels;
        float *test_images;
        float *test_labels;


        float *cur_input;
        float *cur_target;

        EXAMPLE_MNIST_CLASS();
        ~EXAMPLE_MNIST_CLASS();
        void first_read(const char *str);
        void second_init(int batch_size);
        void third_read_train(int index);
};


void get_mnist_image(const char *name, MY_MATRIX_DEVICE *pa);
#endif





