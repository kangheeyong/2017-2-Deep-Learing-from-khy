#include "my_device_func.cuh"

__global__ void make_ones(float *a, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        a[tid] = 1.0;  
        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void make_zeros(float *a, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        a[tid] = 0.0;  
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void transpose(float *dst, const float *str, int str_row, int str_column,int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int x,y;
  
  while(tid < n)
  {
      y = tid % str_row;
      x = tid / str_row;
      dst[IDX2C(x,y,str_column)] = str[IDX2C(y,x,str_row)];
      tid+= blockDim.x * gridDim.x;
  }
}


__global__ void add_bias(const float *a, const float *b, float *c,  int a_row, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        c[tid] = a[tid] + b[tid % a_row];  
        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void transfer(float *dst,const float *str, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        dst[tid] = str[tid];  
        tid+= blockDim.x * gridDim.x;
    }
}



__global__ void relu(float *dst,const float *str, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        if(str[tid] > 0 ) dst[tid] = str[tid];
        else dst[tid] = 0.0;      

        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void relu_inv(float *dst,const float *str, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
        if(str[tid] > 0 ) dst[tid] = 1.0;
        else dst[tid] = 0.0;      
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void scalar_multi(const float *a, const float *b, float *c,  int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
        c[tid] = a[tid]*b[tid];
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void elu(float *dst,const float *str, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        if(str[tid] >= 0 ) dst[tid] = str[tid];
        else dst[tid] = expf(str[tid]) - 1;      


        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void elu_inv(float *dst,const float *str, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
        if(str[tid] >= 0 ) dst[tid] = 1.0;
        else dst[tid] = expf(str[tid]);   
        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void binary_cross_entropy(const float *a,const float *b,float *c, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    while(tid < n)
    {
        c[tid] = -0.5*(b[tid]*logf(a[tid] + 1e-8) + (1-b[tid])*logf(1-a[tid] + 1e-8));
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void binary_cross_entropy_inv(const float *a,const float *b,float *c, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    while(tid < n)
    {
        
        c[tid] = 0.5*(a[tid] - b[tid])/(a[tid]*(1-a[tid]) + 1e-8);
        tid+= blockDim.x * gridDim.x;
    }
}




__global__ void dropout_table(float *a, float dropout_rate,int  n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      if(a[tid] > dropout_rate) a[tid] = 0.0;
      else a[tid] = 1.0; 
      tid+= blockDim.x * gridDim.x;
  }
}


__global__ void dropout(const float *a,const float *b, float *c,float dropout_rate,int n)
{
  long tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      c[tid] = a[tid]*b[tid]/dropout_rate; 
      tid+= blockDim.x * gridDim.x;
  }
}


__global__ void sigmoid(float *dst, const float *str, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        dst[tid] = 1/(1 + expf(-str[tid]));

        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void sigmoid_inv(float *dst,const float *str, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float  y;
 
    while(tid < n)
    {
        y = 1/(1 + expf(-str[tid]));
        dst[tid] = y*(1 - y) + 1e-8;
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void tanh(float *dst,const float *str,int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        dst[tid] = tanhf(str[tid]);

        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void tanh_inv(float *dst,const float *str,int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float  y;
    while(tid < n)
    {
        y = tanhf(str[tid]);
        dst[tid] = (1 + y)*(1 - y);
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void add(const float *a, const float *b, float *c, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        c[tid] = a[tid] + b[tid];  
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void least_squares(const float *a,const float *b,float *c, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
        c[tid] = (a[tid]-b[tid])*(a[tid]-b[tid]);
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void least_squares_inv(const float *a,const float *b,float *c, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        c[tid] = 2*(a[tid]-b[tid]);
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void momentum_vector(const float *a, float *b, float l_rate,float m_rate, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        b[tid] = m_rate*b[tid] - l_rate*a[tid];  
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void adam_beta1(const float *a, float *b, float beta1, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        b[tid] = beta1*b[tid] + (1.0 - beta1)*a[tid];  
        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void adam_beta2(const float *a, float *b, float beta2, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        b[tid] = beta2*b[tid] + (1.0 - beta2)*a[tid]*a[tid];  
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void adam_sum(const float *a, const float *b, float *c, float learning_rate,
        float beta1_t,float beta2_t, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        c[tid] = c[tid] - learning_rate*((a[tid])/(1.0-beta1_t))/(sqrtf(b[tid]/(1.0-beta2_t)) + 1e-8);  
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void max_norm(float *a, float rate, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        if(a[tid] > rate) a[tid] = rate;
        else if(a[tid] < -rate) a[tid] = -rate;
        tid+= blockDim.x * gridDim.x;
    }
}


__global__ void min(float *dst,const float *str, float rate, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    {
        if(str[tid] > rate ) dst[tid] = rate;
        else dst[tid] = str[tid];      

        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void min_inv(float *dst,const float *str,float rate, int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n)
    { 
        if(str[tid] > rate ) dst[tid] = 0.0;
        else dst[tid] = 1.0;      
        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void accuracy_table(const float *y, const float *t,float *r, int row ,int column)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    float max;
    int id_y;
    int id_t;
    while(tid < column)
    {
        max = 0.0;

        for(int i = 0 ; i < row ;i++)
        {
            if(y[IDX2C(i,tid,row)] > max)
            {
                max = y[IDX2C(i,tid,row)];
                id_y = i;
            }
            if(t[IDX2C(i,tid,row)] >= 0.9999) id_t = i;
        }
        if(id_y == id_t) r[tid] = 1.0;
        else r[tid] = 0.0;
        

        tid+= blockDim.x * gridDim.x;
    }
}











