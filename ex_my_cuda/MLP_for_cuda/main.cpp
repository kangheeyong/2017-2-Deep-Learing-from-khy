#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "MLP_basic.h"


using namespace std;

bool ChoseGpuAvailable(int n);

int main(int argc,char **argv)
{
    clock_t t;
    int host2device_time, device2host_time,GPU_time;


    
    int neurals[4] = {3,4,3,2};
    int layers = 4;
    int batch_size = 8;
    float alpha = 0.2;
    float ramda = 0.000001;
    
    
    ChoseGpuAvailable(1);
    
    MLP_basic *a = new MLP_basic;
    a->init(neurals,layers,batch_size,alpha,ramda);
    a->test_example();
   
    t = clock();
    a->cpy_host_device();
    host2device_time = clock() - t;
    
    t = clock();
    a->activation();
    a->delta_rule();
    a->update();
    GPU_time = clock() - t;


    cout<<"host to device time : "<<(double)host2device_time/CLOCKS_PER_SEC<<"s, clock : "<<host2device_time<<endl;
    cout<<"GPU time : "<<(double)GPU_time/CLOCKS_PER_SEC<<"s, clock : "<<GPU_time<<endl;
    cout<<"device to host time : "<<(double)device2host_time/CLOCKS_PER_SEC<<"s, clock : "<<device2host_time<<endl;
 
    delete a;


    return 0;
}




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

