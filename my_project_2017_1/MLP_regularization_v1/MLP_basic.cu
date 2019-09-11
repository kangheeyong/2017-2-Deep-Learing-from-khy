#include "MLP_basic.h"
#include "my_device_function.cuh"
#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column



using namespace std;

MLP_basic:: MLP_basic()
{
    MLP_basic_init();
}

MLP_basic:: MLP_basic(long seed)
{
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_gen,seed));
    MLP_basic_init();
}

void MLP_basic:: MLP_basic_init()
{
    total_layers = 0;
    max_batch = 0;
    alpha = 0;
    ramda = 0;
    beta1 = 0.9;
    beta2 = 0.999;
    beta1_t = beta1;
    beta2_t = beta2;
    
    
    d_target = NULL;
    d_temp = NULL;
    d_temp1 = NULL;
    d_one_vector = NULL;

    d_train_input = NULL;
    d_train_target = NULL;
    d_validation_input = NULL;
    d_validation_target = NULL;
    d_test_input = NULL;
    d_test_target = NULL;   

    for(long i = 0 ; i < MAXIMUM_LAYERS ; i++)
    {
        neural[i] = 0;
        W[i] = NULL;
        b[i] = NULL;

        d_W[i] = NULL;
        d_b[i] = NULL;
        d_a[i] = NULL;
        d_z[i] = NULL;
        d_delta[i] = NULL;
        d_dropout[i] = NULL;
        d_delta_W[i] = NULL;
        d_delta_b[i] = NULL;
        d_adam_W_m[i] = NULL;
        d_adam_W_v[i] = NULL;
        d_adam_b_m[i] = NULL;
        d_adam_b_v[i] = NULL;
    }
    CURAND_CALL(curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT));
    CUBLAS_CALL(cublasCreate(&handle));
    CUDA_CALL(cudaStreamCreate(&stream1)); 
    CUDA_CALL(cudaStreamCreate(&stream2)); 
    CUDA_CALL(cudaStreamCreate(&stream3)); 
    CUDA_CALL(cudaStreamCreate(&stream4)); 

}


MLP_basic :: ~MLP_basic()
{
   if(d_target != NULL) CUDA_CALL(cudaFree(d_target));
   if(d_temp != NULL) CUDA_CALL(cudaFree(d_temp));
   if(d_temp1 != NULL) CUDA_CALL(cudaFree(d_temp1));
   if(d_one_vector != NULL) CUDA_CALL(cudaFree(d_one_vector));

   if(d_train_input != NULL) CUDA_CALL(cudaFree(d_train_input));
   if(d_train_target != NULL) CUDA_CALL(cudaFree(d_train_target));
   if(d_validation_input != NULL) CUDA_CALL(cudaFree(d_validation_input));
   if(d_validation_target != NULL) CUDA_CALL(cudaFree(d_validation_target));
   if(d_test_input != NULL) CUDA_CALL(cudaFree(d_test_input));
   if(d_test_target != NULL) CUDA_CALL(cudaFree(d_test_target));   
   
   for(long i = 0 ; i < MAXIMUM_LAYERS ; i++)
   {
       if(W[i] != NULL) free(W[i]);
       if(b[i] != NULL) free(b[i]);
         
       if(d_W[i] != NULL) CUDA_CALL(cudaFree(d_W[i]));
       if(d_b[i] != NULL) CUDA_CALL(cudaFree(d_b[i]));
       if(d_a[i] != NULL) CUDA_CALL(cudaFree(d_a[i]));
       if(d_z[i] != NULL) CUDA_CALL(cudaFree(d_z[i]));
       if(d_delta[i] != NULL) CUDA_CALL(cudaFree(d_delta[i]));
       if(d_dropout[i] != NULL) CUDA_CALL(cudaFree(d_dropout[i])); 
       if(d_delta_W[i] != NULL) CUDA_CALL(cudaFree(d_delta_W[i]));
       if(d_delta_b[i] != NULL) CUDA_CALL(cudaFree(d_delta_b[i]));
       if(d_adam_W_m[i] != NULL) CUDA_CALL(cudaFree(d_adam_W_m[i]));
       if(d_adam_W_v[i] != NULL) CUDA_CALL(cudaFree(d_adam_W_v[i]));
       if(d_adam_b_m[i] != NULL) CUDA_CALL(cudaFree(d_adam_b_m[i]));
       if(d_adam_b_v[i] != NULL) CUDA_CALL(cudaFree(d_adam_b_v[i])); 
   }

   CUBLAS_CALL(cublasDestroy(handle));
   CURAND_CALL(curandDestroyGenerator(rand_gen));
   CUDA_CALL(cudaStreamDestroy(stream1)); 
   CUDA_CALL(cudaStreamDestroy(stream2)); 
   CUDA_CALL(cudaStreamDestroy(stream3)); 
   CUDA_CALL(cudaStreamDestroy(stream4)); 


}
void MLP_basic :: seed(long seed)
{
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_gen,seed));
}
void MLP_basic :: init(long *neurals,long layers,long max_batch_size,float alpha, float ramda,
        float beta1, float beta2)
{
    this->total_layers = layers;
    this->max_batch = max_batch_size;
    this->alpha = alpha;
    this->ramda = ramda;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->beta1_t = beta1;
    this->beta2_t = beta2;

    
    for(long i = 0 ; i < this->total_layers ; i++)
    {
        this->neural[i] = neurals[i];
    }
 
    CUDA_CALL(cudaMalloc(&d_target,sizeof(float)*neural[total_layers-1]*max_batch));
    CUDA_CALL(cudaMalloc(&d_a[0],sizeof(float)*neural[0]*max_batch));
    
    CUDA_CALL(cudaMalloc(&d_train_input,sizeof(float)*neural[0]*max_batch));   
    CUDA_CALL(cudaMalloc(&d_train_target,sizeof(float)*neural[total_layers - 1]*max_batch));
    CUDA_CALL(cudaMalloc(&d_validation_input,sizeof(float)*neural[0]*max_batch));   
    CUDA_CALL(cudaMalloc(&d_validation_target,sizeof(float)*neural[total_layers - 1]*max_batch));
    CUDA_CALL(cudaMalloc(&d_test_input,sizeof(float)*neural[0]*max_batch));   
    CUDA_CALL(cudaMalloc(&d_test_target,sizeof(float)*neural[total_layers - 1]*max_batch));

    long maximum = 0;
    for(long i = 0 ; i < total_layers-1 ; i++)
    {
        W[i] = (float*)calloc(neural[i]*neural[i+1],sizeof(float));
        b[i] = (float*)calloc(neural[i+1],sizeof(float));

        CUDA_CALL(cudaMalloc(&d_W[i],sizeof(float)*neural[i]*neural[i+1]));
        CUDA_CALL(cudaMalloc(&d_b[i],sizeof(float)*neural[i+1]));
        CUDA_CALL(cudaMalloc(&d_a[i+1],sizeof(float)*neural[i+1]*max_batch));
        CUDA_CALL(cudaMalloc(&d_z[i+1],sizeof(float)*neural[i+1]*max_batch));
        CUDA_CALL(cudaMalloc(&d_delta[i+1],sizeof(float)*neural[i+1]*max_batch));
        CUDA_CALL(cudaMalloc(&d_dropout[i],sizeof(float)*neural[i]*max_batch));
        CUDA_CALL(cudaMalloc(&d_delta_W[i],sizeof(float)*neural[i+1]*neural[i]));
        CUDA_CALL(cudaMalloc(&d_delta_b[i],sizeof(float)*neural[i+1]));
        CUDA_CALL(cudaMalloc(&d_adam_W_m[i],sizeof(float)*neural[i+1]*neural[i]));
        CUDA_CALL(cudaMalloc(&d_adam_W_v[i],sizeof(float)*neural[i+1]*neural[i]));
        CUDA_CALL(cudaMalloc(&d_adam_b_m[i],sizeof(float)*neural[i+1]));
        CUDA_CALL(cudaMalloc(&d_adam_b_v[i],sizeof(float)*neural[i+1]));
        if(neural[i] > maximum) maximum = neural[i];
    } 
    CUDA_CALL(cudaMalloc(&d_temp,sizeof(float)*maximum*max_batch)); //temp alloc
    CUDA_CALL(cudaMalloc(&d_temp1,sizeof(float)*maximum*max_batch));
    CUDA_CALL(cudaMalloc(&d_one_vector,sizeof(float)*max_batch*maximum));
  
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 


    blocksPerGride = (maximum*max_batch + threadsPerBolck -1)/threadsPerBolck;
    init_ones<<<blocksPerGride, threadsPerBolck>>>(d_one_vector,maximum*max_batch);
    for(long i = 0 ; i < total_layers-1 ; i++)
    {
        blocksPerGride = (neural[i]*neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        init_zeros<<<blocksPerGride, threadsPerBolck>>>(d_adam_W_m[i],neural[i]*neural[i+1]);
        blocksPerGride = (neural[i]*neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        init_zeros<<<blocksPerGride, threadsPerBolck>>>(d_adam_W_v[i],neural[i]*neural[i+1]);
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        init_zeros<<<blocksPerGride, threadsPerBolck>>>(d_adam_b_m[i],neural[i+1]);
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        init_zeros<<<blocksPerGride, threadsPerBolck>>>(d_adam_b_v[i],neural[i+1]);
    } 
}



void MLP_basic :: first_parameters_host_device()
{
     for(long i = 0 ; i < total_layers -1 ; i++)
    {
        CUBLAS_CALL(cublasSetMatrix(neural[i+1],neural[i],sizeof(float),W[i],neural[i+1],d_W[i],neural[i+1]));
        CUBLAS_CALL(cublasSetVector(neural[i+1],sizeof(float),b[i],1,d_b[i],1)); 
    }
}
void MLP_basic :: first_random_parameter()
{
     for(long i = 0 ; i < total_layers -1 ; i++)
    {
        CURAND_CALL(curandGenerateNormal(rand_gen,d_W[i],neural[i+1]*neural[i],PARA_MEAN,PARA_STD));
        CURAND_CALL(curandGenerateNormal(rand_gen,d_b[i],neural[i+1],PARA_MEAN,PARA_STD));
    }
}


void MLP_basic :: second_validation_test_set_host_device(float *validation_input, float* validation_target, 
        long validation_batch_size, float *test_input, float *test_target,long test_batch_size)
{
    CUBLAS_CALL(cublasSetMatrix(neural[0],validation_batch_size,sizeof(float),validation_input,neural[0],
                d_validation_input,neural[0]));
    CUBLAS_CALL(cublasSetMatrix(neural[total_layers-1],validation_batch_size,sizeof(float),validation_target,
                neural[total_layers-1],d_validation_target,neural[total_layers-1])); 
    CUBLAS_CALL(cublasSetMatrix(neural[0],test_batch_size,sizeof(float),test_input,neural[0],
                d_test_input,neural[0]));
    CUBLAS_CALL(cublasSetMatrix(neural[total_layers-1],test_batch_size,sizeof(float),test_target,
                neural[total_layers-1],d_test_target,neural[total_layers-1])); 

}
void MLP_basic :: third_train_set_host_device(float *train_input, float *train_target, long train_batch_size)
{
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
//    CUBLAS_CALL(cublasSetMatrix(neural[0],train_batch_size,sizeof(float),train_input,neural[0],d_train_input,neural[0]));
//    CUBLAS_CALL(cublasSetMatrix(neural[total_layers-1],train_batch_size,sizeof(float),train_target,neural[total_layers-1],d_train_target,neural[total_layers-1])); 
  
    CUDA_CALL(cudaMemcpyAsync(d_train_input,train_input,sizeof(float)*neural[0]*train_batch_size,
            cudaMemcpyHostToDevice,stream1));
 
    CUDA_CALL(cudaMemcpyAsync(d_train_target,train_target,sizeof(float)*neural[total_layers-1]*train_batch_size,
            cudaMemcpyHostToDevice,stream2));


    blocksPerGride = (neural[0]*train_batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck,0,stream1>>>
        (d_train_input,d_a[0],neural[0]*train_batch_size);

    blocksPerGride = (neural[total_layers-1]*train_batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck,0,stream2>>>
        (d_train_target,d_target,neural[total_layers-1]*train_batch_size);

}


void MLP_basic :: validataion_setting(long batch_size)
{
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
    
    blocksPerGride = (neural[0]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck,0,stream1>>>(d_validation_input,d_a[0],neural[0]*batch_size);
   
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck,0,stream2>>>(d_validation_target,d_target,neural[total_layers-1]*batch_size);
}
void MLP_basic :: test_setting(long batch_size)
{
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
    
    blocksPerGride = (neural[0]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck,0,stream1>>>(d_test_input,d_a[0],neural[0]*batch_size);
   
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck,0,stream2>>>(d_test_target,d_target,neural[total_layers-1]*batch_size);
}




void MLP_basic :: train_forward_propagation(long batch_size,float *dropout_rate)
{
    float one = 1.0;
    float zero = 0.0;
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
    
    for(int i = 0 ; i < total_layers-1; i++)
    {
        // dropout(i) -> drop out -> dropout(i)
        CURAND_CALL(curandGenerateUniform(rand_gen,d_temp,neural[i]*batch_size));
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        inverted_dropout<<<blocksPerGride, threadsPerBolck>>>(d_dropout[i],d_temp,dropout_rate[i],neural[i]*batch_size);
        
        //a(i) = dropout(i)*a(i)
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_a[i],d_dropout[i],d_a[i],neural[i]*batch_size);
 
        //z(i+1) = w(i)*a(i)
        CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[i+1],batch_size,neural[i],  &one,  
                    d_W[i],neural[i+1],  d_a[i],neural[i],  &zero,  d_z[i+1],neural[i+1]));
        //z(i+1) = z(i+1) + b(i);
        blocksPerGride = (neural[i+1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[i+1],d_b[i],neural[i+1],neural[i+1]*batch_size);
        
        // z(i+1) -> batch_normalizaion -> z(i+1)

        //a(i+1) = F(z(i+1))
        blocksPerGride = (neural[i+1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        if(i == total_layers - 2) //last layer
        {
            sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[i+1],d_z[i+1],neural[i+1]*batch_size);
        }
        else
        {
            relu<<<blocksPerGride, threadsPerBolck>>>(d_a[i+1],d_z[i+1],neural[i+1]*batch_size);
        }


    }
/*
    //last layer

    // dropout(last-2) -> drop out -> dropout(last-2)
    CURAND_CALL(curandGenerateUniform(rand_gen,d_temp,neural[total_layers-2]*batch_size));
    blocksPerGride = (neural[total_layers-2]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    inverted_dropout<<<blocksPerGride, threadsPerBolck>>>(d_dropout[total_layers-2],
            d_temp,dropout_rate[total_layers-2],neural[total_layers-2]*batch_size);

    //a(last-2) = dropout(last-2)*a(last-2)
    blocksPerGride = (neural[total_layers-2]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_a[total_layers-2],d_dropout[total_layers-2],
            d_a[total_layers-2],neural[total_layers-2]*batch_size);

    //z(last-1) = w(last-2)*a(last-2)
    CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[total_layers-1],batch_size,neural[total_layers-2], 
                &one, d_W[total_layers-2],neural[total_layers-1],  d_a[total_layers-2],neural[total_layers-2],
                &zero,  d_z[total_layers-1],neural[total_layers-1]));
    //z(last-1) = z(last-1) + b(last-2);
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[total_layers-1],d_b[total_layers-2],
            neural[total_layers-1],neural[total_layers-1]*batch_size);
    //a(last-1) = F(z(last-1))
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[total_layers-1],
            d_z[total_layers-1],neural[total_layers-1]*batch_size);
*/
}


void MLP_basic :: test_forward_propagation(long batch_size)
{
    float one = 1.0;
    float zero = 0.0;
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
    
    CUDA_CALL(cudaDeviceSynchronize());
    for(int i = 0 ; i < total_layers-2; i++)
    {
        //z(i+1) = w(i)*a(i)
        CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[i+1],batch_size,neural[i],  &one,  
                    d_W[i],neural[i+1],  d_a[i],neural[i],  &zero,  d_z[i+1],neural[i+1]));
        
        //z(i+1) = z(i+1) + b(i);
        blocksPerGride = (neural[i+1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[i+1],d_b[i],neural[i+1],neural[i+1]*batch_size);
        
        // z(i+1) -> batch_normalizaion -> z(i+1)
        

        //a(i+1) = F(z(i+1))
        blocksPerGride = (neural[i+1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        relu<<<blocksPerGride, threadsPerBolck>>>(d_a[i+1],d_z[i+1],neural[i+1]*batch_size);

    }

    //last layer
    //z(last) = w(last-1)*a(last-1)
    CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[total_layers-1],batch_size,neural[total_layers-2], 
                &one, d_W[total_layers-2],neural[total_layers-1],  d_a[total_layers-2],neural[total_layers-2],
                &zero,  d_z[total_layers-1],neural[total_layers-1]));
    //z(last) = z(last) + b(last-1);
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[total_layers-1],d_b[total_layers-2],
            neural[total_layers-1],neural[total_layers-1]*batch_size);
    //a(last) = F(z(last))
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[total_layers-1],
            d_z[total_layers-1],neural[total_layers-1]*batch_size);

}


void MLP_basic :: delta_rule(long batch_size)
{
    float one = 1.0;
    float zero = 0.0;
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 

   

    // temp = (y-T)*(2*batch_size)
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    last_delta_before_transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_a[total_layers-1],
            d_target,batch_size,neural[total_layers-1]*batch_size);      
    //delta4 = transpose(temp)
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta[total_layers-1],d_temp,neural[total_layers-1],batch_size);

    for(int i = total_layers - 2 ; i > 0 ; i--)
    {   
        //delta(i) = delta(i+1)*W(i)
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,batch_size,neural[i],neural[i+1],  &one,  
                    d_delta[i+1],batch_size,  d_W[i],neural[i+1],  &zero,  d_delta[i],batch_size));
        //temp = f_inv(z(i))
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        relu_inv<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_z[i],neural[i]*batch_size);

        //temp = dropout(i).*temp
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_dropout[i],d_temp,neural[i]*batch_size);
 
        
        
        //temp1 = transpose(temp) 
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp1,d_temp,neural[i],batch_size);
        
       
        //delta(i) = delta(i).*temp1
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_delta[i],d_temp1,d_delta[i],neural[i]*batch_size);
    } 
}

float MLP_basic :: get_loss_error(long batch_size)
{
    float result;
    float one = 1.0;
    float zero = 0.0;
    float number = 1.0/(neural[total_layers-1]*batch_size);
    
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
     
    //temp = -0.5*(T*log(y) + (1-T)*log(1-y))
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    loss_cross_entropy<<<blocksPerGride, threadsPerBolck>>>(d_target,d_a[total_layers-1],d_temp,
           neural[total_layers-1],batch_size);
   
    //temp1(y,1) = temp(y,batch_size)*one_vector(batch_size,1) // 세로끼리의 합
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[total_layers-1],1, batch_size,
                &one,  d_temp,neural[total_layers-1],  d_one_vector, batch_size,  &zero,  d_temp1,neural[total_layers-1]));  
           
    //d_temp(1,1) = one_vector(1,y) * temp(y,1)
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,neural[total_layers-1],
                &number,  d_one_vector,1,  d_temp1,neural[total_layers-1],  &zero,  d_temp,1));  
    
    CUBLAS_CALL(cublasGetMatrix(1,1,sizeof(float),d_temp,1,&result,1));
 
    return result;
}

float MLP_basic :: get_accuracy(long batch_size)
{
    float result;
    float one = 1.0;
    float zero = 0.0;
    
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 

    // 배치를 병렬로 해서 출력값의 최대값의 인덱스와 타겟 인덱스 비교후 일치하면 1 아니면 0반환
    blocksPerGride = (batch_size + threadsPerBolck -1)/threadsPerBolck;
    matching<<<blocksPerGride, threadsPerBolck>>>(d_target,d_a[total_layers-1],d_temp1,
            neural[total_layers-1],batch_size);

    //d_temp(스칼라) = temp1*one_vector // 가로 합
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,batch_size,
                &one,  d_temp1,1, d_one_vector, batch_size,  &zero,  d_temp,1));  

    CUBLAS_CALL(cublasGetMatrix(1,1,sizeof(float),d_temp,1,&result,1));

    return result/batch_size;
}
float MLP_basic :: get_sum_square_weight()
{
    float result = 0.0;
    float result1;
  
    float one = 1.0;
    float zero = 0.0;
   
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 

    for(long i = 0 ; i < total_layers-1 ; i++)
    {

        //temp = W.*W
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_W[i],d_W[i],d_temp,neural[i+1]*neural[i]);

        //temp1 = (one_vector)^T*temp // 세로끼리의 합
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[i], neural[i+1],
                    &one,  d_one_vector,1,  d_temp, neural[i+1],  &zero,  d_temp1,1));  
        //d_temp = temp1*one_vector // 가로 합
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,neural[i],
                    &one,  d_temp1,1,  d_one_vector,neural[i],  &zero,  d_temp,1));  

        CUBLAS_CALL(cublasGetMatrix(1,1,sizeof(float),d_temp,1,&result1,1));
        
        result += result1;
    }
    return result;

}


void MLP_basic :: update_gradient_descent(long batch_size, float maxnorm)
{
    float one = 1.0;
    float zero = 0.0;
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 

    for(int i = 0 ; i < total_layers-1 ; i++)
    {
        //temp = a(i)*delta(i+1)
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[i],neural[i+1],batch_size,  &one,
                    d_a[i],neural[i],  d_delta[i+1],batch_size,  &zero,  d_temp,neural[i]));  
        //delta_W(i) = transpose(temp)
        blocksPerGride = (neural[i]*neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[i],d_temp,neural[i],neural[i+1]);

        //delta_b(i) = one_vector^T * delta(i+1)
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[i+1],batch_size,  &one, 
                    d_one_vector,1,  d_delta[i+1],batch_size,  &zero,  d_delta_b[i],1));  

        //Gradient Descent optimizer start
        //W(i) = W(i) - alpha*(delta_W(i) + ramda*W(i)) 
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[i],d_delta_W[i],alpha,ramda,neural[i+1]*neural[i]);
        //max norm constraints
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        maxnorm_constraints<<<blocksPerGride, threadsPerBolck>>>(d_W[i],maxnorm,neural[i+1]*neural[i]);


        //b(i) = b(i) - alpha*transpose(delta_b(i))
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[i],d_delta_b[i],alpha,neural[i+1]);   
        //Gradient Descent eoptimizer end
        }
}



void MLP_basic :: update_adam(long batch_size, float maxnorm)
{
    float one = 1.0;
    float zero = 0.0;
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 

    for(int i = 0 ; i < total_layers-1 ; i++)
    {
        //temp = a(i)*delta(i+1)
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[i],neural[i+1],batch_size,  &one,
                    d_a[i],neural[i],  d_delta[i+1],batch_size,  &zero,  d_temp,neural[i]));  
        //delta_W(i) = transpose(temp)
        blocksPerGride = (neural[i]*neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[i],d_temp,neural[i],neural[i+1]);

        //delta_b(i) = one_vector^T * delta(i+1)
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[i+1],batch_size,  &one, 
                    d_one_vector,1,  d_delta[i+1],batch_size,  &zero,  d_delta_b[i],1));  

        //Adam optimizer start            
        //adam_W_m(i) = (beta1*adam_W_m(i) + (1-beta1)*deta_W(i))
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        adam_mean<<<blocksPerGride, threadsPerBolck>>>(d_adam_W_m[i],d_delta_W[i],beta1,neural[i+1]*neural[i]);

        //adam_W_v(i) = (beta2*adam_W_v(i) + (1-beta2)*deta_W(i).*deta_W(i))  
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        adam_var<<<blocksPerGride, threadsPerBolck>>>(d_adam_W_v[i],d_delta_W[i],beta2,neural[i+1]*neural[i]);
        //temp = (adam_W_m(i)/(1-beta1_t))./(sqrt(adam_W_v/(1-beta2_t)) + 0.00000001)
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        adam_sum<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_adam_W_m[i],d_adam_W_v[i],
                beta1_t,beta2_t,neural[i+1]*neural[i]);
        //W(i) = W(i) - alpha*(temp + ramda*W(i))
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[i],d_temp,alpha,ramda,neural[i+1]*neural[i]);
         //max norm constraints
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        maxnorm_constraints<<<blocksPerGride, threadsPerBolck>>>(d_W[i],maxnorm,neural[i+1]*neural[i]);



        //adam_b_m(i) = (beta1*adam_b_m(i) + (1-beta1)*deta_b(i))   
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        adam_mean<<<blocksPerGride, threadsPerBolck>>>(d_adam_b_m[i],d_delta_b[i],beta1,neural[i+1]);
        //adam_b_v(i) = (beta2*adam_b_v(i) + (1-beta2)*deta_b(i).*deta_b(i))   
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        adam_var<<<blocksPerGride, threadsPerBolck>>>(d_adam_b_v[i],d_delta_b[i],beta2,neural[i+1]);
        //temp = adam_b_m(i)/(1-beta1_t)/(sqrt(adam_b_v/(1-beta2_t)) + 0.00000001)
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        adam_sum<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_adam_b_m[i],d_adam_b_v[i],
                beta1_t,beta2_t,neural[i+1]);  
        //b(i) = b(i) - alpha*adam_b_m(i) 
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[i],d_temp,alpha,neural[i+1]);   
  
        //Adam optimizer end
    }
    beta1_t = beta1_t*beta1;
    beta2_t = beta2_t*beta2; 
}


void MLP_basic :: temp_print()
{


    float aaa[1000000];  
    cublasStatus_t stat;
    int mini_batch = 10;
    int total = 5;
    stat = cublasGetMatrix(neural[total-2],mini_batch,sizeof(float),d_dropout[total-2],neural[total-2],aaa,neural[total-2]);
  
    cout<<stat<<endl;

    for(int y = 0 ; y < neural[total-2] ; y++)
    {
        for(int x = 0 ; x < mini_batch ;x++)
        {
            printf("%1.4f ",aaa[IDX2C(y,x,neural[total-2])]);
        }
        cout<<endl;
    }
    cout<<endl; 
/*
    stat = cublasGetMatrix(neural[total_layers-1],mini_batch,sizeof(float),d_target,neural[total_layers-1],aaa,neural[total_layers-1]);
  
    cout<<stat<<endl;

    for(int y = 0 ; y < neural[total_layers-1] ; y++)
    {
        for(int x = 0 ; x < mini_batch ;x++)
        {
            printf("%1.4f ",aaa[IDX2C(y,x,neural[total_layers-1])]);
        }
        cout<<endl;
    }
    cout<<endl; 

    int i = 2;
    cout<<"WW: ";
 cout<<cublasGetMatrix(neural[i+1],neural[i],sizeof(float),d_W[i],neural[i+1],aaa,neural[i+1])<<endl;


    for(int y = 0 ; y < neural[i+1] ; y++)
    {
        for(int x = 0 ; x < neural[i] ;x++)
        {
            printf("%1.8f ",aaa[IDX2C(y,x,neural[i+1])]);
        }
        cout<<endl;
    }
    cout<<endl; 


    cout<<beta1_t<<endl;
    cout<<beta1<<endl;

/*    
    
    float aaa[1000000];  
    cublasStatus_t stat;

    int idx = 0;

    stat = cublasGetMatrix(neural[idx],mini_batch,sizeof(float),d_a[idx],neural[idx],aaa,neural[idx]);
  
    cout<<stat<<endl;

    for(int y = 0 ; y < neural[idx] ; y++)
    {
        for(int x = 0 ; x < mini_batch ;x++)
        {
            cout<<aaa[IDX2C(y,x,neural[idx])]<<" ";
        }
        cout<<endl;
    }
    cout<<endl; 
*/

/*
    float aaa[1000000];  
    cublasStatus_t stat;
//    curandStatus_t
    int idx = 2;

    stat = cublasGetMatrix(neural[idx+1],neural[idx],sizeof(float),d_W[idx],neural[idx+1],aaa,neural[idx+1]);
  
    cout<<stat<<endl;

    for(int y = 0 ; y < neural[idx+1] ; y++)
    {
        for(int x = 0 ; x < neural[idx] ;x++)
        {
            cout<<aaa[IDX2C(y,x,neural[idx+1])]<<" ";
        }
        cout<<endl;
    }
    cout<<endl; 
*/

}





















