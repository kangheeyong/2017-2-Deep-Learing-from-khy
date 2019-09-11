#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <pthread.h>
#include "MLP_basic.h"
#include "my_mnist_class.h"


class My_Timer
{
    private :
        struct timespec begin;
        struct timespec end;

    public :
        void start()
        {
            clock_gettime(CLOCK_MONOTONIC, &begin);
        }
        double finish()
        {
            clock_gettime(CLOCK_MONOTONIC, &end);
            return  (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        }
};


using namespace std;

bool ChoseGpuAvailable(int n);
void* thread_function(void *id);




pthread_t thread_handles; 
pthread_mutex_t mutex;
int sync_stat = 0;



My_Timer t1,t2,t3;
double host2device_time, device2host_time,GPU_time;
double setting_time;
double read_time;
double total_time;




EXAMPLE_MNIST_CLASS data;
MLP_basic network;

int neurals[4] = {784,512,128,10};
int layers = 4;
int batch_size = 128;
float alpha = 0.2;
float ramda = 0.000001;
int idx = 0;
unsigned long max_iteration = 10000;


int main(int argc,char **argv)
{
    t3.start();

    ChoseGpuAvailable(1);
    pthread_mutex_init(&mutex,NULL);

    t1.start(); // start setting

    data.first_read("../../MNIST_data");
    data.second_init(batch_size);  
    data.third_read_train(0);
    
    network.init(neurals,layers,batch_size,alpha,ramda);
    network.first_random_parameter();

    setting_time =  t1.finish(); // finish setting


     
    pthread_create(&thread_handles,NULL,thread_function,(void*)NULL);





    for(int iter = 1 ; iter <= max_iteration ; iter++){
        pthread_mutex_lock(&mutex);
        idx = iter%(55000 - batch_size);
        sync_stat = 1;
        pthread_mutex_unlock(&mutex);

    // t2.start(); // start 
    // data.third_read_train(idx);
    // read_time =  t2.finish(); // finish 
           

        t1.start();// timer start

        network.third_train_set_host_device(data.cur_input, data.cur_target); 

        host2device_time = t1.finish(); //timer finish

        t1.start(); //timer start

        //network.temp_print();
        network.activation();
        network.delta_rule();
        network.update();
        
        GPU_time = t1.finish(); //timer finish

    }
    
    pthread_cancel(thread_handles); // thread sync

    
    total_time = t3.finish();

    cout<<"setting_time : "<<setting_time<<endl;
    cout<<"read_time  : "<<read_time<<endl;


    cout<<"host to device time : "<<host2device_time<<endl;
    cout<<"GPU time : "<<GPU_time<<endl;
    cout<<"device to host time : "<<device2host_time<<endl;

    cout<<"total time : "<<total_time<<endl;



    return 0;
}

void* thread_function(void *id)
{
    while(1)
    {
        pthread_mutex_lock(&mutex);
        if(sync_stat == 1)
        {
            t2.start(); // start 
            data.third_read_train(idx);
            read_time =  t2.finish(); // finish 
            sync_stat = 0;
        }
        pthread_mutex_unlock(&mutex);
    }
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

