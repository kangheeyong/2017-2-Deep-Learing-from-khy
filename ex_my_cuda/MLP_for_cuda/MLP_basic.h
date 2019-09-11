#ifndef __MLP_BASIC_CU__
#define __MLP_BASIC_CU__

#define MAXIMUM_LAYERS        1024

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

class MLP_basic
{
    private :
        int total_layers;
        int neural[MAXIMUM_LAYERS];
        int mini_batch;

        float alpha; //learning rate
        float ramda; //decay rate

        float *input; 
        float *target;
 
        float *W[MAXIMUM_LAYERS];
        float *b[MAXIMUM_LAYERS];

        float *d_W[MAXIMUM_LAYERS];
        float *d_b[MAXIMUM_LAYERS];
        float *d_a[MAXIMUM_LAYERS];
        float *d_z[MAXIMUM_LAYERS];
        float *d_delta[MAXIMUM_LAYERS];
        float *d_delta_W[MAXIMUM_LAYERS];
        float *d_delta_b[MAXIMUM_LAYERS];
        float *d_target;
        float *d_temp;
        float *d_temp1;
        float *d_one_vector;
       
        cublasHandle_t handle;
    
   
    public :

        MLP_basic();
        ~MLP_basic();

        void init(int *neurals,int layers,int batch_size,float alpha, float ramda);
        void test_example();
        void cpy_host_device();
        void activation();
        void delta_rule();
        void update();


};




#endif



