#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

//보통 행렬은 3*4 행렬이면 
//
// a11 a12 a13 a14
// a21 a22 a23 a24
// a31 a32 a33 a34
//
// a11 a12 a13 a14 a21 a22 a23 a24 a31 a32 a33 a34 
// 위와 같이 저장하지만
//
// cuBLAS에서는
// 
// a11 a21 a31 a12 a22 a32 a13 a23 a33 a14 a24 a34
// 위와 같이 저장된다.
//
// 보통 열(세로,column)의 수  기준으로 저장하지만
// cuBLAS에서는 행(가로, row)의 수 기준으로 저장한다.
//

#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column

using namespace std;


bool ChoseGpuAvailable(int n)
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);


    cout<<"devicesCount : "<<devicesCount<<endl;
    
    for(int i = 0 ; i < devicesCount ; i++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties,i);
        cout<<"----- device "<<i<<" -----"<<endl;
        cout<<"device name : "<<deviceProperties.name<<endl;
        cout<<"maxThreadsPerBlock : "<<deviceProperties.maxThreadsPerBlock<<endl;
        cout<<"warpSize : "<<deviceProperties.warpSize<<endl;

    }
    if(n > devicesCount && n < 0) return false;
    else
    {
        cudaSetDevice(n);

        return true;
    }
}

__global__ void my_kernel(int m, int k, int n, float *a,float *b, float *c)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int x,y;
    
  while(tid < m*n)
  {
      y = tid % n;
      x = tid / n;
      c[IDX2C(x,y,m)] = 0.0;
      for(int i = 0 ; i < k ;i++)
      {
          c[IDX2C(x,y,m)] += a[IDX2C(y,i,n)]*b[IDX2C(i,x,k)];
      }
      tid+= blockDim.x * gridDim.x;
  }
}



int main(int argc, char** argv)
{

    unsigned int m = 6; //row
    unsigned int n = 4; //column
    unsigned int k = 5;
    float *matrix1, *matrix2, *matrix3;
    clock_t t;
    int host2device_time, device2host_time,GPU_time;

    matrix1 = new float[n*k];
    matrix2 = new float[k*m];
    matrix3 = new float[m*n];
    
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    ChoseGpuAvailable(1);
    
     //데이터 초기화
   
    int ind = 11;

    for(int x = 0 ; x < k ; x++)
    {
        for(int y = 0 ; y < n ; y++)
        {
            matrix1[IDX2C(y,x,n)] = (float)ind++;
            
        }
    }
    ind = 11;
    for(int x = 0 ; x < m ; x++)
    {
        for(int y = 0 ; y < k ; y++)
        {
            matrix2[IDX2C(y,x,k)] = (float)ind++;
            
        }
    }
    ind = 11;
    for(int x = 0 ; x < n ; x++)
    {
        for(int y = 0 ; y < m ; y++)
        {
            matrix3[IDX2C(y,x,m)] = (float)ind++;
            
        }
    }



    cout<<"maxtrix1"<<endl;
    for(int y = 0 ; y < n ; y++)
    {
        for(int x = 0 ; x < k ; x++)
        {
            cout<<matrix1[IDX2C(y,x,n)]<<" "; 
            
        }
        cout<<endl;
    }
     cout<<"maxtrix2"<<endl;
    for(int y = 0 ; y < k ; y++)
    {
        for(int x = 0 ; x < m ; x++)
        {
            cout<<matrix2[IDX2C(y,x,k)]<<" "; 
            
        }
        cout<<endl;
    }
    cout<<"maxtrix3"<<endl;
    for(int y = 0 ; y < m ; y++)
    {
        for(int x = 0 ; x < n ; x++)
        {
            cout<<matrix3[IDX2C(y,x,m)]<<" "; 
            
        }
        cout<<endl;
    }
    
    cout<<endl;


    //cuda 메모리 할당
    float *d_matrix1, *d_matrix2, *d_matrix3;
    cudaMalloc(&d_matrix1,n*k*sizeof(float));
    cudaMalloc(&d_matrix2,k*m*sizeof(float));
    cudaMalloc(&d_matrix3,m*n*sizeof(float));

    // memory -> cuda memory
    t = clock();
    cublasCreate(&handle);
    cublasSetMatrix(n,k,sizeof(float),matrix1,n,d_matrix1,n);
    cublasSetMatrix(k,m,sizeof(float),matrix2,k,d_matrix2,k);
    cublasSetMatrix(m,n,sizeof(float),matrix3,m,d_matrix3,m);
    host2device_time = clock()-t;


    // 연산 (커널 실행)
    float al=1.0f;
    float bet=0.0f;
    t = clock();
    //stat = cublasSgemv(handle,  CUBLAS_OP_N,m,n,  &al,  d_matrix1,m,  d_vector1,1,  &bet,  d_vector2,1);
    //
    //먼저 주소 값의 바꿔주면서 행렬의 (0,0)의 위치를 바꿔주고
    //3,4번째 파라메터로 행렬의 최종 크기를 정해준다.
    //
    //CUBLAS_OP_N은 아무것도 안한것
    //CUBLAS_OP_T는 transpose한것이다.
    //
    //
    int threadsPerBolck = 1024;
    int blocksPerGride = (m*n + threadsPerBolck -1)/threadsPerBolck;
    my_kernel<<<blocksPerGride, threadsPerBolck>>>(m,k,n,d_matrix1,d_matrix2,d_matrix3);
  

    GPU_time = clock() - t;
    //cuda memory -> memory
    
    t= clock();
     
    cublasGetMatrix(n,k,sizeof(float),d_matrix1,n,matrix1,n);
    cublasGetMatrix(k,m,sizeof(float),d_matrix2,k,matrix2,k);
    cublasGetMatrix(m,n,sizeof(float),d_matrix3,m,matrix3,m);
 
    device2host_time = clock() - t; 
     //결과 확인
    

    cout<<"maxtrix1"<<endl;
    for(int y = 0 ; y < n ; y++)
    {
        for(int x = 0 ; x < k ; x++)
        {
            cout<<matrix1[IDX2C(y,x,n)]<<" "; 
            
        }
        cout<<endl;
    }
     cout<<"maxtrix2"<<endl;
    for(int y = 0 ; y < k ; y++)
    {
        for(int x = 0 ; x < m ; x++)
        {
            cout<<matrix2[IDX2C(y,x,k)]<<" "; 
            
        }
        cout<<endl;
    }
    cout<<"maxtrix3"<<endl;
    for(int y = 0 ; y < m ; y++)
    {
        for(int x = 0 ; x < n ; x++)
        {
            cout<<matrix3[IDX2C(y,x,m)]<<" "; 
            
        }
        cout<<endl;
    }
    

    
   
    cout<<"host to device time : "<<host2device_time<<endl;
    cout<<"GPU time : "<<GPU_time<<endl;
    cout<<"device to host time : "<<device2host_time<<endl;
   
    //cuda 메모리 해제
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix3);
    
    cublasDestroy(handle);

    delete matrix1;
    delete matrix2;
    delete matrix3;


    return 0;
}

