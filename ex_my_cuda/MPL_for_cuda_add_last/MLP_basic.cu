#include "MLP_basic.h"

#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column



using namespace std;


float GausianRandom(float average, float stdev) 
{
    double v1, v2, s, temp;

    do {
        v1 =  2 * ((float) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        v2 =  2 * ((float) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    s = sqrt( (-2 * log(s)) / s );

    temp = v1 * s;
    temp =( stdev*temp) + average;


    return temp;
}

MLP_basic:: MLP_basic()
{
    total_layers = 0;
    max_batch = 0;
    alpha = 0;
    ramda = 0;
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
        d_delta_W[i] = NULL;
        d_delta_b[i] = NULL;

    }
  
    CURAND_CALL(curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT));


    CUBLAS_CALL(cublasCreate(&handle));
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
       if(d_delta_W[i] != NULL) CUDA_CALL(cudaFree(d_delta_W[i]));
       if(d_delta_b[i] != NULL) CUDA_CALL(cudaFree(d_delta_b[i]));
   }
   CUBLAS_CALL(cublasDestroy(handle));
   CURAND_CALL(curandDestroyGenerator(rand_gen));

}

void MLP_basic :: init(long *neurals,long layers,long max_batch_size,float alpha, float ramda)
{
    this->total_layers = layers;
    this->max_batch = max_batch_size;
    this->alpha = alpha;
    this->ramda = ramda;


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
        CUDA_CALL(cudaMalloc(&d_delta_W[i],sizeof(float)*neural[i+1]*neural[i]));
        CUDA_CALL(cudaMalloc(&d_delta_b[i],sizeof(float)*neural[i+1]));
        if(neural[i] > maximum) maximum = neural[i];
    } 
    CUDA_CALL(cudaMalloc(&d_temp,sizeof(float)*maximum*max_batch)); //temp alloc
    CUDA_CALL(cudaMalloc(&d_temp1,sizeof(float)*maximum*max_batch));
    
    float *one_vector;
    one_vector = (float*)calloc(max_batch,sizeof(float));
    for(long i = 0 ; i < max_batch ; i++) one_vector[i] = 1.0;
    
    CUDA_CALL(cudaMalloc(&d_one_vector,sizeof(float)*max_batch*maximum));
    CUBLAS_CALL(cublasSetMatrix(1,max_batch,sizeof(float),one_vector,1,d_one_vector,1));  
    
    free(one_vector);

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

__global__ void deliver_front_to_rear(float *front,float *rear,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
 
  while(tid < n)
  {
      rear[tid] = front[tid];  
      tid+= blockDim.x * gridDim.x;
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
   
    
    CUBLAS_CALL(cublasSetMatrix(neural[0],train_batch_size,sizeof(float),train_input,neural[0],d_train_input,neural[0]));
    CUBLAS_CALL(cublasSetMatrix(neural[total_layers-1],train_batch_size,sizeof(float),train_target,neural[total_layers-1],d_train_target,neural[total_layers-1])); 

    blocksPerGride = (neural[0]*train_batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck>>>(d_train_input,d_a[0],neural[0]*train_batch_size);
   
    blocksPerGride = (neural[total_layers-1]*train_batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck>>>(d_train_target,d_target,neural[total_layers-1]*train_batch_size);

}


void MLP_basic :: validataion_setting(long batch_size)
{
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
   
    
    blocksPerGride = (neural[0]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck>>>(d_validation_input,d_a[0],neural[0]*batch_size);
   
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck>>>(d_validation_target,d_target,neural[total_layers-1]*batch_size);
}
void MLP_basic :: test_setting(long batch_size)
{
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
   
    
    blocksPerGride = (neural[0]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck>>>(d_test_input,d_a[0],neural[0]*batch_size);
   
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    deliver_front_to_rear<<<blocksPerGride, threadsPerBolck>>>(d_test_target,d_target,neural[total_layers-1]*batch_size);
}


__global__ void add_bias(float *z,float *b,long column,long n)
{
    long tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        z[tid] += b[tid % column];  

        tid+= blockDim.x * gridDim.x;
    }
}
__global__ void sigmoid(float *a,float *z,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      a[tid] = 1/(1+expf(-z[tid]));
      
      tid+= blockDim.x * gridDim.x;
  }
}
__global__ void F(float *a,float *z,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      //      if(z[tid] > 0 ) a[tid] = z[tid];
      //      else a[tid] = 0.0;      
      a[tid] = 1/(1+expf(-z[tid]));


      tid+= blockDim.x * gridDim.x;
  }
}





void MLP_basic :: activation(long batch_size)
{
    float one = 1.0;
    float zero = 0.0;
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
    
    for(int i = 0 ; i < total_layers-2; i++)
    {
        CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[i+1],batch_size,neural[i],  &one,  
                    d_W[i],neural[i+1],  d_a[i],neural[i],  &zero,  d_z[i+1],neural[i+1]));
        //z2 = z2 + b1;
        blocksPerGride = (neural[i+1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[i+1],d_b[i],neural[i+1],neural[i+1]*batch_size);
        //a2 = F(z2)
        blocksPerGride = (neural[i+1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        F<<<blocksPerGride, threadsPerBolck>>>(d_a[i+1],d_z[i+1],neural[i+1]*batch_size);

    }
/*
    //z2 = W1*a1
    CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[1],batch_size,neural[0],  &one,  d_W[0],neural[1],  d_a[0],neural[0],  &zero,  d_z[1],neural[1]));
    //z2 = z2 + b1;
    blocksPerGride = (neural[1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[1],d_b[0],neural[1],neural[1]*batch_size);
    //a2 = F(z2)
    blocksPerGride = (neural[1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[1],d_z[1],neural[1]*batch_size);
    //

    //z3 = W2*a2
    CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[2],batch_size,neural[1],  &one,  d_W[1],neural[2],  d_a[1],neural[1],  &zero,  d_z[2],neural[2]));
    //z3 = z3 + b2;
    blocksPerGride = (neural[2]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[2],d_b[1],neural[2],neural[2]*batch_size);
    //a3 = F(z3)
    blocksPerGride = (neural[2]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[2],d_z[2],neural[2]*batch_size);
    //
    
    //z4 = W3*a3
    CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[3],batch_size,neural[2],  &one,  d_W[2],neural[3],  d_a[2],neural[2],  &zero,  d_z[3],neural[3]));
    //z4 = z4 + b3;
    blocksPerGride = (neural[3]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[3],d_b[2],neural[3],neural[3]*batch_size);
    //a4 = F(z4)
    blocksPerGride = (neural[3]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[3],d_z[3],neural[3]*batch_size);
*/
    CUBLAS_CALL(cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[total_layers-1],batch_size,neural[total_layers-2], 
                &one, d_W[total_layers-2],neural[total_layers-1],  d_a[total_layers-2],neural[total_layers-2],
                &zero,  d_z[total_layers-1],neural[total_layers-1]));
    //z4 = z4 + b3;
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[total_layers-1],d_b[total_layers-2],
            neural[total_layers-1],neural[total_layers-1]*batch_size);
    //a4 = F(z4)
    blocksPerGride = (neural[total_layers-1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[total_layers-1],
            d_z[total_layers-1],neural[total_layers-1]*batch_size);





}

__global__ void last_delta_before_transpose(float *temp, float *y,float *T,long batch_size,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      temp[tid] = (y[tid]-T[tid])/(2*batch_size);   
      tid+= blockDim.x * gridDim.x;
  }
}
__global__ void transpose(float *after, float *before,long before_columns,long before_rows)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  long x,y;
  
  while(tid < before_columns*before_rows)
  {
      y = tid % before_columns;
      x = tid / before_columns;
      after[IDX2C(x,y,before_rows)] = before[IDX2C(y,x,before_columns)];
      tid+= blockDim.x * gridDim.x;
  }
}

__global__ void sigmoid_inv(float *a,float *z,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      a[tid] = (1/(1+expf(-z[tid])))*(1 - 1/(1+expf(-z[tid])));
      tid+= blockDim.x * gridDim.x;
  }
}

__global__ void F_inv(float *a,float *z,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      a[tid] = (1/(1+expf(-z[tid])))*(1 - 1/(1+expf(-z[tid])));
      tid+= blockDim.x * gridDim.x;
  }
}

__global__ void basic_multi(float *a,float *b,float *c, long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      c[tid] = a[tid]*b[tid]; 
      tid+= blockDim.x * gridDim.x;
  }
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
/* 
    // temp = (y-T)*(2*batch_size)
    blocksPerGride = (neural[3]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    last_delta_before_transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_a[3],d_target,batch_size,neural[3]*batch_size);      
    //delta4 = transpose(temp)
    blocksPerGride = (neural[3]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta[3],d_temp,neural[3],batch_size);
    
   
  
    //delta3 = delta4*W3
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,batch_size,neural[2],neural[3],  &one,  d_delta[3],batch_size,  d_W[2],neural[3],  &zero,  d_delta[2],batch_size));  
    //temp = f_inv(z3)
    blocksPerGride = (neural[2]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid_inv<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_z[2],neural[2]*batch_size);   
    //temp1 = transpose(temp) 
    blocksPerGride = (neural[2]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp1,d_temp,neural[2],batch_size);
    //delta3 = delta3.*temp1
    blocksPerGride = (neural[2]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_delta[2],d_temp1,d_delta[2],neural[2]*batch_size);
    

    //delta2 = delta3*W2
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,batch_size,neural[1],neural[2],  &one,  d_delta[2],batch_size,  d_W[1],neural[2],  &zero,  d_delta[1],batch_size));
    //temp = f_inv(z2)
    blocksPerGride = (neural[1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    sigmoid_inv<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_z[1],neural[1]*batch_size);
    //temp1 = transpose(temp) 
    blocksPerGride = (neural[1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp1,d_temp,neural[1],batch_size);
    //delta2 = delta2.*temp1
    blocksPerGride = (neural[1]*batch_size + threadsPerBolck -1)/threadsPerBolck;
    basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_delta[1],d_temp1,d_delta[1],neural[1]*batch_size);
  */  

    for(int i = total_layers - 2 ; i > 0 ; i--)
    {   
        //delta2 = delta3*W2
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,batch_size,neural[i],neural[i+1],  &one,  
                    d_delta[i+1],batch_size,  d_W[i],neural[i+1],  &zero,  d_delta[i],batch_size));
        //temp = f_inv(z2)
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        F_inv<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_z[i],neural[i]*batch_size);
        //temp1 = transpose(temp) 
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp1,d_temp,neural[i],batch_size);
        //delta2 = delta2.*temp1
        blocksPerGride = (neural[i]*batch_size + threadsPerBolck -1)/threadsPerBolck;
        basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_delta[i],d_temp1,d_delta[i],neural[i]*batch_size);
    } 
}


__global__ void loss_cross_entropy(float *target,float *y, float * result,long last_neural,long batch_size)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
 
  while(tid < last_neural*batch_size)
  {
      result[tid] = -0.5*(target[tid]*logf(y[tid] + 0.000000001) + (1.0 - target[tid])*logf(1-y[tid] + 0.000000001));      
      tid+= blockDim.x * gridDim.x;
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

__global__ void matching(float *target,float *y, float * result,long last_neural,long batch_size)
{
    long tid = blockIdx.x*blockDim.x + threadIdx.x;

    int target_inx;
    int y_inx;

    while(tid < batch_size)
    {
        float max = 0.0;
        for(int i = 0 ; i < last_neural ; i++)
        {
            if(target[IDX2C(i,tid,last_neural)] > 0.9)
            {
                target_inx = i;
            }
            if(y[IDX2C(i,tid,last_neural)] > max)
            {
                max = y[IDX2C(i,tid,last_neural)];
                y_inx = i;
            }
        }
        if(target_inx == y_inx) result[tid] = 1.0;
        else result[tid] = 0.0;

        tid+= blockDim.x * gridDim.x;
    }
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



__global__ void weight_update(float *w,float *delta_w, float alpha,float ramda,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
 
  while(tid < n)
  {
      w[tid] = w[tid] - alpha*(delta_w[tid] + ramda*w[tid]);    
      tid+= blockDim.x * gridDim.x;
  }
}
__global__ void bias_update(float *b,float *delta_b, float alpha,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
 
  while(tid < n)
  {
      b[tid] = b[tid] - alpha*delta_b[tid];    
      tid+= blockDim.x * gridDim.x;
  }
}




void MLP_basic :: update(long batch_size)
{
    float one = 1.0;
    float zero = 0.0;
    long threadsPerBolck = 1024;
    long blocksPerGride = 0; 
    
    for(int i = 0 ; i < total_layers-1 ; i++)
    {
        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[i],neural[i+1],batch_size,  &one,
                    d_a[i],neural[i],  d_delta[i+1],batch_size,  &zero,  d_temp,neural[i]));  
        //delta_W1 = transpose(temp)
        blocksPerGride = (neural[i]*neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[i],d_temp,neural[i],neural[i+1]);
        //W1 = W1 - alpha*(delta_W1 + ramda*W1) 
        blocksPerGride = (neural[i+1]*neural[i] + threadsPerBolck -1)/threadsPerBolck;
        weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[i],d_delta_W[i],alpha,ramda,neural[i+1]*neural[i]);


        CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[i+1],batch_size,  &one,
                    d_one_vector,1,  d_delta[i+1],batch_size,  &zero,  d_delta_b[i],1));  
        //b1 = b1 - alpha*transpose(delta_b1)
        blocksPerGride = (neural[i+1] + threadsPerBolck -1)/threadsPerBolck;
        bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[i],d_delta_b[i],alpha,neural[i+1]);   



    }

    /*
    //temp = a3*delta4
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[2],neural[3],batch_size,  &one, 
    d_a[2],neural[2],  d_delta[3],batch_size,  &zero,  d_temp,neural[2]));  
    //delta_W3 = transpose(temp)
    blocksPerGride = (neural[2]*neural[3] + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[2],d_temp,neural[2],neural[3]);
    //W3 = W3 - alpha*(delta_W3 + ramda*W3) 
    blocksPerGride = (neural[3]*neural[2] + threadsPerBolck -1)/threadsPerBolck;
    weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[2],d_delta_W[2],alpha,ramda,neural[3]*neural[2]);   
  
    //temp = a2*delta3
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[1],neural[2],batch_size,  &one,  d_a[1],neural[1],  d_delta[2],batch_size,  &zero,  d_temp,neural[1]));  
    //delta_W2 = transpose(temp)
    blocksPerGride = (neural[1]*neural[2] + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[1],d_temp,neural[1],neural[2]);
    //W2 = W2 - alpha*(delta_W2 + ramda*W2) 
    blocksPerGride = (neural[2]*neural[1] + threadsPerBolck -1)/threadsPerBolck;
    weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[1],d_delta_W[1],alpha,ramda,neural[2]*neural[1]);
   
    //temp = a1*delta2

    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[0],neural[1],batch_size,  &one,  d_a[0],neural[0],  d_delta[1],batch_size,  &zero,  d_temp,neural[0]));  
    //delta_W1 = transpose(temp)
    blocksPerGride = (neural[0]*neural[1] + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[0],d_temp,neural[0],neural[1]);
    //W1 = W1 - alpha*(delta_W1 + ramda*W1) 
    blocksPerGride = (neural[1]*neural[0] + threadsPerBolck -1)/threadsPerBolck;
    weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[0],d_delta_W[0],alpha,ramda,neural[1]*neural[0]);
    
    
    
    //delta_b3 = one_vector*delta4
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[3],batch_size,  &one,  d_one_vector,1,  d_delta[3],batch_size,  &zero,  d_delta_b[2],1));  
    //b3 = b3 - alpha*transpose(delta_b3)
    blocksPerGride = (neural[3] + threadsPerBolck -1)/threadsPerBolck;
    bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[2],d_delta_b[2],alpha,neural[3]);   

    //delta_b2 = one_vector*delta3
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[2],batch_size,  &one,  d_one_vector,1,  d_delta[2],batch_size,  &zero,  d_delta_b[1],1));  
    //b2 = b2 - alpha*transpose(delta_b2)
    blocksPerGride = (neural[2] + threadsPerBolck -1)/threadsPerBolck;
    bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[1],d_delta_b[1],alpha,neural[2]);   

    //delta_b1 = one_vector*delta2
    CUBLAS_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[1],batch_size,  &one,  d_one_vector,1,  d_delta[1],batch_size,  &zero,  d_delta_b[0],1));  
    //b1 = b1 - alpha*transpose(delta_b1)
    blocksPerGride = (neural[1] + threadsPerBolck -1)/threadsPerBolck;
    bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[0],d_delta_b[0],alpha,neural[1]);   
*/
}


void MLP_basic :: temp_print()
{


    float aaa[1000000];  
    cublasStatus_t stat;
    int mini_batch = 10;

    stat = cublasGetMatrix(neural[total_layers-1],mini_batch,sizeof(float),d_a[total_layers-1],neural[total_layers-1],aaa,neural[total_layers-1]);
  
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





















