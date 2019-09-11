#ifndef __MLP_BASIC_CU__
#define __MLP_BASIC_CU__

#define MAXIMUM_LAYERS        1024
#define PARA_MEAN             0.0
#define PARA_STD              0.1

#define MAX_VALIDATION        5000
#define MAX_TEST              10000


#define CUDA_CALL(x) do{if((x) != cudaSuccess){\
    printf("Error at %s:%d",__FILE__,__LINE__);\
    return EXIT_FSILURE;}}while(0)

#define CUBLAS_CALL(x) do{if((x) != CUBLAS_STATUS_SUCCESS){\
    printf("Error at %s:%d",__FILE__,__LINE__);\
    return EXIT_FSILURE;}}while(0)


#define CURAND_CALL(x) do{if((x) != CURAND_STATUS_SUCCESS){\
    printf("Error at %s:%d",__FILE__,__LINE__);\
    return EXIT_FSILURE;}}while(0)





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
        int total_layers;
        int neural[MAXIMUM_LAYERS];
        int mini_batch;

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

        void init(int *neurals,int layers,int batch_size,float alpha, float ramda);
        
        void first_parameters_host_device();
        void first_random_parameter();
        void second_validation_test_set_host_device(float *validataion_input, float* validataion_target, float *test_input, float *test_target);
        void third_train_set_host_device(float *train_input, float *train_target); 
        
        void activation();
        void delta_rule();
        void update();

        void validataion_activation();
        void test_activation();

        void temp_print();
};




#endif



