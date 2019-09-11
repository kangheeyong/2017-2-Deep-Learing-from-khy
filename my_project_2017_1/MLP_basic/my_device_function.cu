#include "my_device_function.cuh"



__global__ void sigmoid(float *a,float *z,long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      a[tid] = 1/(1+expf(-z[tid]));
      
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

__global__ void relu(float *a,float *z,long n)
{
    long tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        if(z[tid] > 0 ) a[tid] = z[tid];
        else a[tid] = 0.0;      

        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void relu_inv(float *a,float *z,long n)
{
    long tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
        if(z[tid] > 0 ) a[tid] = 1.0;
        else a[tid] = 0.0;      
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void new_activation(float *a,float *z,long n)
{
 long tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
//        if(z[tid] > 0 ) a[tid] = 1.0;
//        else a[tid] = 0.0;      
        tid+= blockDim.x * gridDim.x;
    }

}
__global__ void new_activation_inv(float *a,float *z,long n)
{
 long tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
//        if(z[tid] > 0 ) a[tid] = 1.0;
//        else a[tid] = 0.0;      
        tid+= blockDim.x * gridDim.x;
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


__global__ void add_bias(float *z,float *b,long column,long n)
{
    long tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        z[tid] += b[tid % column];  

        tid+= blockDim.x * gridDim.x;
    }
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
__global__ void basic_multi(float *a,float *b,float *c, long n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      c[tid] = a[tid]*b[tid]; 
      tid+= blockDim.x * gridDim.x;
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







