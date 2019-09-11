#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>


#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column
#define N                   1000000000



using namespace std;




__global__ void my_kernel(float *a, float *b, float *c)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  while(tid < N)
  {
      c[tid] = b[tid] + a[tid];

      tid+= blockDim.x * gridDim.x;
  }
}

int main(int argc, char** argv)
{
    int vecter_n = N;
    float *a, *b, *c;
    clock_t t;
    a = new float[vecter_n];
    b = new float[vecter_n];
    c = new float[vecter_n];
    
     //데이터 초기화
    for(int i = 0 ; i < vecter_n ; i++)
    {
        a[i] = i- 48.5;
        b[i] = i+vecter_n-48.5;
        c[i] = 0;
    //    printf("a[%d] : %0.3f, bi[%d] : %0.3f, c[%d] : %0.3f\n",i,a[i],i,b[i],i,c[i]);
    }
    cout<<"before"<<endl;
    
    //cuda 메모리 할당
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a,vecter_n*sizeof(float));
    cudaMalloc(&d_b,vecter_n*sizeof(float));
    cudaMalloc(&d_c,vecter_n*sizeof(float));

    // memory -> cuda memory

    cublasSetVector(vecter_n,sizeof(float),a,1,d_a,1);
    cublasSetVector(vecter_n,sizeof(float),b,1,d_b,1);
    cublasSetVector(vecter_n,sizeof(float),c,1,d_c,1);



    // 연산 (커널 실행)
    int threadsPerBolck = 256;
    int blocksPerGride = (vecter_n + threadsPerBolck -1)/threadsPerBolck;
    t = clock();
    my_kernel<<<blocksPerGride, threadsPerBolck>>>(d_a,d_b,d_c);
    t = clock() - t;

    //cuda memory -> memory
    
    cublasGetVector(vecter_n,sizeof(float),d_a,1,a,1);
    cublasGetVector(vecter_n,sizeof(float),d_b,1,b,1);
    cublasGetVector(vecter_n,sizeof(float),d_c,1,c,1);
    

     //결과 확인
    for(int i = 0 ; i < vecter_n ; i++)
    {
      //  printf("a[%d] : %0.3f, b[%d] : %0.3f, c[%d] : %0.3f\n",i,a[i],i,b[i],i,c[i]);
    }
    cout<<"time : "<<t<<endl;
    //cuda 메모리 해제
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    


    delete a;
    delete b;
    delete c;


    return 0;
}

