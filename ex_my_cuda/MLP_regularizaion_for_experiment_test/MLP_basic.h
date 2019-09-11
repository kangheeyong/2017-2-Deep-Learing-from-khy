#ifndef __MLP_BASIC_CU__
#define __MLP_BASIC_CU__

#define MAXIMUM_LAYERS        1024
#define PARA_MEAN             0.0
#define PARA_STD              0.01



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

class MLP_basic
{
    private :
        long total_layers;
        long neural[MAXIMUM_LAYERS];
        long max_batch;

        float alpha; //learning rate
        float ramda; //decay rate
        float beta1;
        float beta2;

        float beta1_t;
        float beta2_t;
 
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
        float *d_dropout[MAXIMUM_LAYERS];
        float *d_delta_W[MAXIMUM_LAYERS];
        float *d_delta_b[MAXIMUM_LAYERS];
        float *d_adam_W_m[MAXIMUM_LAYERS];
        float *d_adam_W_v[MAXIMUM_LAYERS];
        float *d_adam_b_m[MAXIMUM_LAYERS];
        float *d_adam_b_v[MAXIMUM_LAYERS];
        float *d_target;
        float *d_temp;
        float *d_temp1;
        float *d_one_vector;
       
        cublasHandle_t handle;
        curandGenerator_t rand_gen;
        cudaStream_t stream1,stream2,stream3,stream4; 
        void MLP_basic_init();
    public :

        MLP_basic();
        MLP_basic(long seed);
        ~MLP_basic();

        void seed(long seed); 
        void init(long *neurals,long layers,long max_batch_size,float alpha, float ramda,
                float beta1, float beta2);
        
        void first_parameters_host_device();
        void first_random_parameter();
        void second_validation_test_set_host_device(float *validation_input, float* validation_target, 
                long validation_batch_size, float *test_input, float *test_target, long test_batch_size);
        void third_train_set_host_device(float *train_input, float *train_target,long train_batch_size); 
        
        void train_forward_propagation(long batch_size, float *dropout_rate);
        void test_forward_propagation(long batch_size);
        void delta_rule(long batch_size);
        void update_gradient_descent(long batch_size, float maxnorm);
        void update_adam(long batch_size, float maxnorm);
        
        float get_loss_error(long batch_size);
        float get_accuracy(long batch_size);
        float get_sum_square_weight();

        void validataion_setting(long batch_size);
        void test_setting(long batch_size);

        void temp_print();
};




#endif



