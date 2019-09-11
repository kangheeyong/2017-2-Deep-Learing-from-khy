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
    mini_batch = 0;
    alpha = 0;
    ramda = 0;
    target = NULL;
    d_target = NULL;
    d_temp = NULL;
    d_temp1 = NULL;
    d_one_vector = NULL;
    input = NULL; 
    for(int i = 0 ; i < MAXIMUM_LAYERS ; i++)
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

    cublasCreate(&handle);
}

MLP_basic :: ~MLP_basic()
{
   if(target != NULL) free(target);
   if(d_target != NULL) cudaFree(d_target);
   if(d_temp != NULL) cudaFree(d_temp);
   if(d_temp1 != NULL) cudaFree(d_temp1);
   if(d_one_vector != NULL) cudaFree(d_one_vector);
   if(input != NULL) free(input);

   for(int i = 0 ; i < MAXIMUM_LAYERS ; i++)
   {
       if(W[i] != NULL) free(W[i]);
       if(b[i] != NULL) free(b[i]);
         
       if(d_W[i] != NULL) cudaFree(d_W[i]);
       if(d_b[i] != NULL) cudaFree(d_b[i]);
       if(d_a[i] != NULL) cudaFree(d_a[i]);
       if(d_z[i] != NULL) cudaFree(d_z[i]);
       if(d_delta[i] != NULL) cudaFree(d_delta[i]);
       if(d_delta_W[i] != NULL) cudaFree(d_delta_W[i]);
       if(d_delta_b[i] != NULL) cudaFree(d_delta_b[i]);
   }
   cublasDestroy(handle);
    
    
}

void MLP_basic :: init(int *neurals,int layers,int batch_size,float alpha, float ramda)
{
    this->total_layers = layers;
    this->mini_batch = batch_size;
    this->alpha = alpha;
    this->ramda = ramda;
    
    for(int i = 0 ; i < this->total_layers ; i++)
    {
        this->neural[i] = neurals[i];
    }

    cudaMalloc(&d_target,sizeof(float)*neural[total_layers-1]*mini_batch);
    cudaMalloc(&d_a[0],sizeof(float)*neural[0]*mini_batch);

    target = (float*)calloc(neural[total_layers-1]*mini_batch,sizeof(float));
    input = (float*)calloc(neural[0]*mini_batch,sizeof(float));
    

    int maximum = 0;
    for(int i = 0 ; i < total_layers-1 ; i++)
    {
        W[i] = (float*)calloc(neural[i]*neural[i+1],sizeof(float));
        b[i] = (float*)calloc(neural[i+1],sizeof(float));

        cudaMalloc(&d_W[i],sizeof(float)*neural[i]*neural[i+1]);
        cudaMalloc(&d_b[i],sizeof(float)*neural[i+1]);
        cudaMalloc(&d_a[i+1],sizeof(float)*neural[i+1]*mini_batch);
        cudaMalloc(&d_z[i+1],sizeof(float)*neural[i+1]*mini_batch);
        cudaMalloc(&d_delta[i+1],sizeof(float)*neural[i+1]*mini_batch);
        cudaMalloc(&d_delta_W[i],sizeof(float)*neural[i+1]*neural[i]);
        cudaMalloc(&d_delta_b[i],sizeof(float)*neural[i+1]);
        if(neural[i] > maximum) maximum = neural[i]; 
    } 
    cudaMalloc(&d_temp,sizeof(float)*maximum*maximum); //temp alloc
    cudaMalloc(&d_temp1,sizeof(float)*maximum*maximum);
    
    float *one_vector;
    one_vector = (float*)calloc(mini_batch,sizeof(float));
    for(int i = 0 ; i < mini_batch ; i++) one_vector[i] = 1.0;
    
    cudaMalloc(&d_one_vector,sizeof(float)*mini_batch);
    cublasSetMatrix(1,mini_batch,sizeof(float),one_vector,1,d_one_vector,1);  
    
    free(one_vector);

}

void MLP_basic :: test_example()
{
   input[0] = 0;
    input[1] = 0;
    input[2] = 0;
    input[3] = 0;
    input[4] = 0;
    input[5] = 1;
    input[6] = 0;
    input[7] = 1;
    input[8] = 0;
    input[9] = 0;
    input[10] = 1;
    input[11] = 1;
    input[12] = 1;
    input[13] = 0;
    input[14] = 0;
    input[15] = 1;
    input[16] = 0;
    input[17] = 1;
    input[18] = 1;
    input[19] = 1;
    input[20] = 0;
    input[21] = 1;
    input[22] = 1;
    input[23] = 1;

    target[0] = 1;
    target[1] = 1;
    target[2] = 1;
    target[3] = 0;
    target[4] = 0;
    target[5] = 0;
    target[6] = 0;
    target[7] = 1;
    target[8] = 0;
    target[9] = 1;
    target[10] = 0;
    target[11] = 0;
    target[12] = 1;
    target[13] = 0;
    target[14] = 1;
    target[15] = 1;


        
    W[0][0] = -0.2052;
    W[0][1] = 0.3735;
    W[0][2] = -0.2398;
    W[0][3] = -0.3509;
    W[0][4] = -0.2674;
    W[0][5] = -0.1811;
    W[0][6] = -0.1767;
    W[0][7] = 0.0564;
    W[0][8] = -0.0623;
    W[0][9] = 0.1854;
    W[0][10] = -0.2694;
    W[0][11] = 0.2531;
   
    W[1][0] = -0.6215;
    W[1][1] = -0.4016;
    W[1][2] = 0.3716;
    W[1][3] = 0.6417;
    W[1][4] = -0.0870;
    W[1][5] = 0.1909;
    W[1][6] = 0.5633;
    W[1][7] = 0.5609;
    W[1][8] = -0.5898;
    W[1][9] = -0.3324;
    W[1][10] = -0.1320;
    W[1][11] = 0.4434;

    W[2][0] = 0.0209;
    W[2][1] = 0.1427;
    W[2][2] = -0.1019;
    W[2][3] = 0.2493;
    W[2][4] = 0.2937;
    W[2][5] = -0.1552;

    b[0][0] = -0.4773;
    b[0][1] = 0.2770;
    b[0][2] = -0.0768;
    b[0][3] = 0.0528;

    b[1][0] = -0.4325;
    b[1][1] = -0.0925;
    b[1][2] = 0.4019;

    b[2][0] = 0.5683;
    b[2][1] = 0.1868;


}

void MLP_basic :: cpy_host_device()
{

    for(int i = 0 ; i < total_layers -1 ; i++)
    {
        cublasSetMatrix(neural[i+1],neural[i],sizeof(float),W[i],neural[i+1],d_W[i],neural[i+1]);
        cublasSetVector(neural[i+1],sizeof(float),b[i],1,d_b[i],1); 
    }

    //test code
    cublasSetMatrix(neural[0],mini_batch,sizeof(float),input,neural[0],d_a[0],neural[0]);    
    cublasSetMatrix(neural[total_layers-1],mini_batch,sizeof(float),target,neural[total_layers-1],d_target,neural[total_layers-1]);
}


__global__ void add_bias(float *z,float *b,int column,int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      z[tid] += b[tid % column];  
      
      tid+= blockDim.x * gridDim.x;
  }
}
__global__ void sigmoid(float *a,float *z,int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      a[tid] = 1/(1+expf(-z[tid]));
      
      tid+= blockDim.x * gridDim.x;
  }
}




void MLP_basic :: activation()
{
    float one = 1.0;
    float zero = 0.0;
    int threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    


    //z2 = W1*a1
    cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[1],mini_batch,neural[0],  &one,  d_W[0],neural[1],  d_a[0],neural[0],  &zero,  d_z[1],neural[1]);
    //z2 = z2 + b1;
    blocksPerGride = (neural[1]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[1],d_b[0],neural[1],neural[1]*mini_batch);
    //a2 = F(z2)
    blocksPerGride = (neural[1]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[1],d_z[1],neural[1]*mini_batch);
    //

    //z3 = W2*a2
    cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[2],mini_batch,neural[1],  &one,  d_W[1],neural[2],  d_a[1],neural[1],  &zero,  d_z[2],neural[2]);
    //z3 = z3 + b2;
    blocksPerGride = (neural[2]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[2],d_b[1],neural[2],neural[2]*mini_batch);
    //a3 = F(z3)
    blocksPerGride = (neural[2]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[2],d_z[2],neural[2]*mini_batch);
    //
    
    //z4 = W3*a3
    cublasSgemm(handle,  CUBLAS_OP_N,CUBLAS_OP_N,neural[3],mini_batch,neural[2],  &one,  d_W[2],neural[3],  d_a[2],neural[2],  &zero,  d_z[3],neural[3]);
    //z4 = z4 + b3;
    blocksPerGride = (neural[3]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    add_bias<<<blocksPerGride, threadsPerBolck>>>(d_z[3],d_b[2],neural[3],neural[3]*mini_batch);
    //a4 = F(z4)
    blocksPerGride = (neural[3]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    sigmoid<<<blocksPerGride, threadsPerBolck>>>(d_a[3],d_z[3],neural[3]*mini_batch);
    
/*
    cublasGetMatrix(neural[3],mini_batch,sizeof(float),d_a[3],neural[3],target,neural[3]);
    
    for(int y = 0 ; y < neural[3] ; y++)
    {
        for(int x = 0 ; x < mini_batch ;x++)
        {
            cout<<target[IDX2C(y,x,neural[3])]<<" ";
        }
        cout<<endl;
    }
    */
}

__global__ void last_delta_before_transpose(float *temp, float *y,float *T,int batch_size,int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      temp[tid] = (y[tid]-T[tid])/(2*batch_size);   
      tid+= blockDim.x * gridDim.x;
  }
}
__global__ void transpose(float *after, float *before,int before_columns,int before_rows)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int x,y;
  
  while(tid < before_columns*before_rows)
  {
      y = tid % before_columns;
      x = tid / before_columns;
      after[IDX2C(x,y,before_rows)] = before[IDX2C(y,x,before_columns)];
      tid+= blockDim.x * gridDim.x;
  }
}

__global__ void sigmoid_inv(float *a,float *z,int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      a[tid] = (1/(1+expf(-z[tid])))*(1 - 1/(1+expf(-z[tid])));
      tid+= blockDim.x * gridDim.x;
  }
}

__global__ void basic_multi(float *a,float *b,float *c, int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < n)
  {
      c[tid] = a[tid]*b[tid]; 
      tid+= blockDim.x * gridDim.x;
  }
}



void MLP_basic :: delta_rule()
{
    float one = 1.0;
    float zero = 0.0;
    int threadsPerBolck = 1024;
    int blocksPerGride = 0; 

   

    // temp = (y-T)*(2*batch_size)
    blocksPerGride = (neural[3]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    last_delta_before_transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_a[3],d_target,mini_batch,neural[3]*mini_batch);      
    //delta4 = transpose(temp)
    blocksPerGride = (neural[3]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta[3],d_temp,neural[3],mini_batch);
    
  
    //delta3 = delta4*W3
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,mini_batch,neural[2],neural[3],  &one,  d_delta[3],mini_batch,  d_W[2],neural[3],  &zero,  d_delta[2],mini_batch);  
    //temp = f_inv(z3)
    blocksPerGride = (neural[2]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    sigmoid_inv<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_z[2],neural[2]*mini_batch);   
    //temp1 = transpose(temp) 
    blocksPerGride = (neural[2]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp1,d_temp,neural[2],mini_batch);
    //delta3 = delta3.*temp1
    blocksPerGride = (neural[2]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_delta[2],d_temp1,d_delta[2],neural[2]*mini_batch);
    

    //delta2 = delta3*W2
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,mini_batch,neural[1],neural[2],  &one,  d_delta[2],mini_batch,  d_W[1],neural[2],  &zero,  d_delta[1],mini_batch);
    //temp = f_inv(z2)
    blocksPerGride = (neural[1]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    sigmoid_inv<<<blocksPerGride, threadsPerBolck>>>(d_temp,d_z[1],neural[1]*mini_batch);
    //temp1 = transpose(temp) 
    blocksPerGride = (neural[1]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_temp1,d_temp,neural[1],mini_batch);
    //delta2 = delta2.*temp1
    blocksPerGride = (neural[1]*mini_batch + threadsPerBolck -1)/threadsPerBolck;
    basic_multi<<<blocksPerGride, threadsPerBolck>>>(d_delta[1],d_temp1,d_delta[1],neural[1]*mini_batch);
    
      /* 
    
    float aaa[10000];
    cublasGetMatrix(mini_batch,neural[1],sizeof(float),d_delta[1],mini_batch,aaa,mini_batch);
    
    for(int y = 0 ; y < mini_batch ; y++)
    {
        for(int x = 0 ; x < neural[1] ;x++)
        {
            cout<<aaa[IDX2C(y,x,mini_batch)]<<" ";
        }
        cout<<endl;
    }
   cout<<endl; 
    */
      
}



__global__ void weight_update(float *w,float *delta_w, float alpha,float ramda,int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
 
  while(tid < n)
  {
      w[tid] = w[tid] - alpha*(delta_w[tid] + ramda*w[tid]);    
      tid+= blockDim.x * gridDim.x;
  }
}
__global__ void bias_update(float *b,float *delta_b, float alpha,int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
 
  while(tid < n)
  {
      b[tid] = b[tid] - alpha*delta_b[tid];    
      tid+= blockDim.x * gridDim.x;
  }
}




void MLP_basic :: update()
{
    float one = 1.0;
    float zero = 0.0;
    int threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    

    
    //temp = a3*delta4
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[2],neural[3],mini_batch,  &one,  d_a[2],neural[2],  d_delta[3],mini_batch,  &zero,  d_temp,neural[2]);  
    //delta_W3 = transpose(temp)
    blocksPerGride = (neural[2]*neural[3] + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[2],d_temp,neural[2],neural[3]);
    //W3 = W3 - alpha*(delta_W3 + ramda*W3) 
    blocksPerGride = (neural[3]*neural[2] + threadsPerBolck -1)/threadsPerBolck;
    weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[2],d_delta_W[2],alpha,ramda,neural[3]*neural[2]);   
   
    //temp = a2*delta3
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[1],neural[2],mini_batch,  &one,  d_a[1],neural[1],  d_delta[2],mini_batch,  &zero,  d_temp,neural[1]);  
    //delta_W2 = transpose(temp)
    blocksPerGride = (neural[1]*neural[2] + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[1],d_temp,neural[1],neural[2]);
    //W2 = W2 - alpha*(delta_W2 + ramda*W2) 
    blocksPerGride = (neural[2]*neural[1] + threadsPerBolck -1)/threadsPerBolck;
    weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[1],d_delta_W[1],alpha,ramda,neural[2]*neural[1]);
    
    //temp = a1*delta2
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,neural[0],neural[1],mini_batch,  &one,  d_a[0],neural[0],  d_delta[1],mini_batch,  &zero,  d_temp,neural[0]);  
    //delta_W1 = transpose(temp)
    blocksPerGride = (neural[0]*neural[1] + threadsPerBolck -1)/threadsPerBolck;
    transpose<<<blocksPerGride, threadsPerBolck>>>(d_delta_W[0],d_temp,neural[0],neural[1]);
    //W1 = W1 - alpha*(delta_W1 + ramda*W1) 
    blocksPerGride = (neural[1]*neural[0] + threadsPerBolck -1)/threadsPerBolck;
    weight_update<<<blocksPerGride, threadsPerBolck>>>(d_W[0],d_delta_W[0],alpha,ramda,neural[1]*neural[0]);
  
    
    
    //delta_b3 = one_vector*delta4
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[3],mini_batch,  &one,  d_one_vector,1,  d_delta[3],mini_batch,  &zero,  d_delta_b[2],1);  
    //b3 = b3 - alpha*transpose(delta_b3)
    blocksPerGride = (neural[3] + threadsPerBolck -1)/threadsPerBolck;
    bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[2],d_delta_b[2],alpha,neural[3]);   

    //delta_b2 = one_vector*delta3
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[2],mini_batch,  &one,  d_one_vector,1,  d_delta[2],mini_batch,  &zero,  d_delta_b[1],1);  
    //b2 = b2 - alpha*transpose(delta_b2)
    blocksPerGride = (neural[2] + threadsPerBolck -1)/threadsPerBolck;
    bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[1],d_delta_b[1],alpha,neural[2]);   

    //delta_b1 = one_vector*delta2
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,neural[1],mini_batch,  &one,  d_one_vector,1,  d_delta[1],mini_batch,  &zero,  d_delta_b[0],1);  
    //b1 = b1 - alpha*transpose(delta_b1)
    blocksPerGride = (neural[1] + threadsPerBolck -1)/threadsPerBolck;
    bias_update<<<blocksPerGride, threadsPerBolck>>>(d_b[0],d_delta_b[0],alpha,neural[1]);   

/*
    float bbb[10000];  
    cublasStatus_t stat; 
    stat = cublasGetMatrix(1,neural[1],sizeof(float),d_b[0],1,bbb,1);
  
    cout<<stat<<endl;

    for(int y = 0 ; y < mini_batch ; y++)
    {
        cout<<bbb[y]<<" ";
    }
    cout<<endl; 

*/


/*  
    float aaa[10000];  
    cublasStatus_t stat; 
    stat = cublasGetMatrix(neural[1],neural[0],sizeof(float),d_W[0],neural[1],aaa,neural[1]);
  
    cout<<stat<<endl;

    for(int y = 0 ; y < neural[1] ; y++)
    {
        for(int x = 0 ; x < neural[0] ;x++)
        {
            cout<<aaa[IDX2C(y,x,neural[1])]<<" ";
        }
        cout<<endl;
    }
    cout<<endl; 

 */
   


}























