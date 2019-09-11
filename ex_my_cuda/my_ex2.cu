#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>


#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column
#define N                   1000000



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


__global__ void my_kernel(float *a, float *b, float *c)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < N)
  {
      //c[tid] = rintf( a[tid]);
      
     if(a[tid] > 0) c[tid] = logf(a[tid] + 1);
     else c[tid] = -logf(-a[tid] + 1);
      
      tid+= blockDim.x * gridDim.x;
  }
}

int main(int argc, char** argv)
{
    int vecter_n = N;
    float *a, *b, *c;
    clock_t t;
    int host2device_time, device2host_time,GPU_time;

    a = new float[vecter_n];
    b = new float[vecter_n];
    c = new float[vecter_n];
    ChoseGpuAvailable(1);
    
     //데이터 초기화
    for(int i = 0 ; i < vecter_n ; i++)
    {
        a[i] = i- 48.5;
        b[i] = i+vecter_n-48.5;
        c[i] = 0;
   //     printf("a[%d] : %0.3f, bi[%d] : %0.3f, c[%d] : %0.3f\n",i,a[i],i,b[i],i,c[i]);
    }
    cout<<"----- before -----"<<endl;
    
    //cuda 메모리 할당
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a,vecter_n*sizeof(float));
    cudaMalloc(&d_b,vecter_n*sizeof(float));
    cudaMalloc(&d_c,vecter_n*sizeof(float));

    // memory -> cuda memory
    t = clock();
    cublasSetVector(vecter_n,sizeof(float),a,1,d_a,1);
    cublasSetVector(vecter_n,sizeof(float),b,1,d_b,1);
    cublasSetVector(vecter_n,sizeof(float),c,1,d_c,1);
    host2device_time = clock()-t;


    // 연산 (커널 실행)
    int threadsPerBolck = 1024;
    int blocksPerGride = (vecter_n + threadsPerBolck -1)/threadsPerBolck;
    
    t = clock();
    my_kernel<<<blocksPerGride, threadsPerBolck>>>(d_a,d_b,d_c);
    my_kernel<<<blocksPerGride, threadsPerBolck>>>(d_a,d_b,d_c);
    my_kernel<<<blocksPerGride, threadsPerBolck>>>(d_a,d_b,d_c);
    my_kernel<<<blocksPerGride, threadsPerBolck>>>(d_a,d_b,d_c);




    GPU_time = clock() - t;
    
    //cuda memory -> memory
    t= clock();
    cublasGetVector(vecter_n,sizeof(float),d_a,1,a,1);
    cublasGetVector(vecter_n,sizeof(float),d_b,1,b,1);
    cublasGetVector(vecter_n,sizeof(float),d_c,1,c,1);
    device2host_time = clock() - t;

    //결과 확인
    for(int i = 0 ; i < vecter_n ; i++)
    {
   //     printf("a[%d] : %0.3f, b[%d] : %0.3f, c[%d] : %0.3f\n",i,a[i],i,b[i],i,c[i]);
    }

    cout<<"host to device time : "<<host2device_time<<endl;
    cout<<"GPU time : "<<GPU_time<<endl;
    cout<<"device to host time : "<<device2host_time<<endl;
   
    //cuda 메모리 해제
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    


    delete a;
    delete b;
    delete c;


    return 0;
}

