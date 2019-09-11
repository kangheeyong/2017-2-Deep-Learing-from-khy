#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>


#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column
#define N                   1000000000



using namespace std;




void my_kernel(float *a, float *b, float *c)
{
  int tid = 0;
  while(tid < N)
  {
      c[tid] = b[tid] + a[tid];

      tid++;;
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
   //     printf("a[%d] : %0.3f, b[%d] : %0.3f, c[%d] : %0.3f\n",i,a[i],i,b[i],i,c[i]);
    }
    cout<<"before"<<endl;
    


    // 연산 (커널 실행)
    t = clock();
    my_kernel(a,b,c);
    t = clock() - t;

    

     //결과 확인
    for(int i = 0 ; i < vecter_n ; i++)
    {
     //   printf("a[%d] : %0.3f, b[%d] : %0.3f, c[%d] : %0.3f\n",i,a[i],i,b[i],i,c[i]);
    }
    cout<<"time : "<<t<<endl;
    


    delete a;
    delete b;
    delete c;


    return 0;
}

