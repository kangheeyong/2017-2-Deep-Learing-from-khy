#ifndef __MLP_BASIC_CU__
#define __MLP_BASIC_CU__

#define MAXIMUM_LAYERS        1024
#define PARA_MEAN             0.0
#define PARA_STD              0.1



#define CUDA_CALL(x)          if((x) != cudaSuccess){\
                              printf("CUDA Error at %s:%d\n",__FILE__,__LINE__);\
                              exit(0);}

#define CUBLAS_CALL(x)        if((x) != CUBLAS_STATUS_SUCCESS){\
                              printf("CUBLAS Error at %s:%d\n",__FILE__,__LINE__);\
                              exit(0);}

#define CURAND_CALL(x)        if((x) != CURAND_STATUS_SUCCESS){\
                              printf("CURAND Error at %s:%d\n",__FILE__,__LINE__);\
                              exit(0);}


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>

class MLP_basic
{
    private :
        long total_layers;
        long neural[MAXIMUM_LAYERS];
        long max_batch;

        float alpha; //learning rate
        float ramda; //decay rate

 
        float *W[MAXIMUM_LAYERS];
        float *b[MAXIMUM_LAYERS];

        float *d_train_input;
        float *d_train_target;
        float *d_validation_input;
        float *d_validation_target;
        float *d_test_input;
        float *d_test_target;

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
        curandGenerator_t rand_gen;
     
   
    public :

        MLP_basic();
        ~MLP_basic();

        void init(long *neurals,long layers,long max_batch_size,float alpha, float ramda);
        
        void first_parameters_host_device();
        void first_random_parameter();
        void second_validation_test_set_host_device(float *validation_input, float* validation_target, 
                long validation_batch_size, float *test_input, float *test_target, long test_batch_size);
        void third_train_set_host_device(float *train_input, float *train_target,long train_batch_size); 
        
        void activation(long batch_size);
        void delta_rule(long batch_size);
        void update(long batch_size);
        
        float get_loss_error(long batch_size);
        float get_accuracy(long batch_size);
        float get_sum_square_weight();

        void validataion_setting(long batch_size);
        void test_setting(long batch_size);

        void temp_print();
};




#endif



