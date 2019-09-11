#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>


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



int main(int argc, char** argv)
{

    unsigned int m = 60000;
    unsigned int n = 50000;

    float *matrix1, *vector1, *vector2;
    clock_t t;
    int host2device_time, device2host_time,GPU_time;

    matrix1 = new float[m*n];
    vector1 = new float[n];
    vector2 = new float[m];
    
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    ChoseGpuAvailable(1);
    
     //데이터 초기화
   
    int ind = 11;

    for(int j = 0 ; j < n ; j++)
    {
        for(int i = 0 ; i < m ; i++)
        {
            matrix1[IDX2C(i,j,m)] = (float)ind++;
            
        }
    }
    for(int i = 0 ; i < n ; i++) vector1[i] = 1.0f;
    for(int i = 0 ; i < m ; i++) vector2[i] = 0.0f;
/*
    cout<<"maxtrix1"<<endl;
    for(int j = 0 ; j < n ; j++)
    {
        for(int i = 0 ; i < m ; i++)
        {
            cout<<matrix1[IDX2C(i,j,m)]<<" "; 
            
        }
        cout<<endl;
    }
    cout<<"[vector1]^T"<<endl;
    for(int i = 0 ; i < n ; i++) cout<<vector1[i]<<" ";
    cout<<endl;
    cout<<"[vector2]^T"<<endl;
    for(int i = 0 ; i < m ; i++) cout<<vector2[i]<<" ";
    cout<<endl;
*/

    //cuda 메모리 할당
    float *d_matrix1, *d_vector1, *d_vector2;
    cudaMalloc(&d_matrix1,n*m*sizeof(float));
    cudaMalloc(&d_vector1,n*sizeof(float));
    cudaMalloc(&d_vector2,m*sizeof(float));

    // memory -> cuda memory
    t = clock();
    cublasCreate(&handle);
    cublasSetMatrix(m,n,sizeof(float),matrix1,m,d_matrix1,m);
    cublasSetVector(n,sizeof(float),vector1,1,d_vector1,1);
    cublasSetVector(m,sizeof(float),vector2,1,d_vector2,1);
    host2device_time = clock()-t;


    // 연산 (커널 실행)
    float al=1.0f;
    float bet=1.0f;
    t = clock();
    stat = cublasSgemv(handle,  CUBLAS_OP_N,m,n,  &al,  d_matrix1,m,  d_vector1,1,  &bet,  d_vector2,1);
    stat = cublasSgemv(handle,  CUBLAS_OP_N,m,n,  &al,  d_matrix1,m,  d_vector1,1,  &bet,  d_vector2,1);


    GPU_time = clock() - t;
    
    //cuda memory -> memory
    t= clock();
    
    cublasGetMatrix(m,n,sizeof(float),d_matrix1,m,matrix1,m);
    cublasGetVector(n,sizeof(float),d_vector1,1,vector1,1);
    cublasGetVector(m,sizeof(float),d_vector2,1,vector2,1);
    device2host_time = clock() - t;

    //결과 확인
 /*   cout<<"maxtrix1"<<endl;
    for(int j = 0 ; j < n ; j++)
    {
        for(int i = 0 ; i < m ; i++)
        {
            cout<<matrix1[IDX2C(i,j,m)]<<" "; 
            
        }
        cout<<endl;
    }
    cout<<"[vector1]^T"<<endl;
    for(int i = 0 ; i < n ; i++) cout<<vector1[i]<<" ";
    cout<<endl;
    cout<<"[vector2]^T"<<endl;
    for(int i = 0 ; i < m ; i++) cout<<vector2[i]<<" ";
    cout<<endl;
*/

    cout<<"host to device time : "<<host2device_time<<endl;
    cout<<"GPU time : "<<GPU_time<<endl;
    cout<<"device to host time : "<<device2host_time<<endl;
   
    //cuda 메모리 해제
    cudaFree(d_matrix1);
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    
    cublasDestroy(handle);

    delete matrix1;
    delete vector1;
    delete vector2;


    return 0;
}

