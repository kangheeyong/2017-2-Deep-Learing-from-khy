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
bool sync_stat = false;


My_Timer t1,t2,t3;
double host2device_time = 0.0;
double device2host_time = 0.0;
double GPU_time = 0.0;
double setting_time =0.0;
double read_time =0.0;
double total_time =0.0;


float train_error,train_accuracy;
float validation_error,validation_accuracy;
float test_error, test_accuracy;


EXAMPLE_MNIST_CLASS data;
MLP_basic network;


long neurals[4] = {784,2024,2024,10};
long layers = 4;

long train_batch_size = 128;
long validation_batch_size = 5000;
long test_batch_size = 10000;
long maximum_train_set = 55000;


float alpha = 0.0001;       //learning rate
float ramda = 0.00000001;   //weight decay para
float beta1 = 0.9;          //adam para
float beta2 = 0.999;        //adam paea

long idx = 0;
unsigned long max_iteration = 100000;
unsigned long print_per_iteration = 10000;


int main(int argc,char **argv)
{
    t3.start();

    ChoseGpuAvailable(0);
    pthread_mutex_init(&mutex,NULL);

    t1.start(); // start setting

    data.first_read("../../MNIST_data");
    data.second_init(train_batch_size);  
    data.third_read_train(0);
    
    long max_batch_size;
    max_batch_size = (validation_batch_size > test_batch_size)? validation_batch_size : test_batch_size;
    max_batch_size = (max_batch_size > train_batch_size)? max_batch_size : train_batch_size;
    network.init(neurals,layers,max_batch_size,alpha,ramda,beta1,beta2);
    
    network.first_random_parameter();
    network.second_validation_test_set_host_device(data.validation_images, data.validation_labels, 
                validation_batch_size, data.test_images, data.test_labels, test_batch_size);
    setting_time =  t1.finish(); // finish setting

     
    pthread_create(&thread_handles,NULL,thread_function,(void*)NULL);
    
    float *temp_input, *temp_target;
    for(long iter = 1 ; iter <= max_iteration; iter++){
        
        pthread_mutex_lock(&mutex);
        idx = iter%(maximum_train_set - train_batch_size);
        sync_stat = true;
        temp_input = data.cur_input;
        temp_target = data.cur_target;
        pthread_mutex_unlock(&mutex);


        t1.start();// timer start
        network.third_train_set_host_device(temp_input, temp_target,train_batch_size); 
        host2device_time += t1.finish(); //timer finish

        

        t1.start(); //timer start
        network.forward_propagation(train_batch_size);
        //train_error = network.get_loss_error(train_batch_size);   
        //train_accuracy = network.get_accuracy(train_batch_size);
        //network.get_sum_square_weight();
        network.delta_rule(train_batch_size);
        network.update_adam(train_batch_size);
       // network.temp_print();
       // network.validataion_setting(validation_batch_size);
       // network.activation(validation_batch_size);
       // validation_error = network.get_loss_error(validation_batch_size);   
       // validation_accuracy = network.get_accuracy(validation_batch_size);

        GPU_time += t1.finish(); //timer finish

        if(iter% print_per_iteration == 0)
        {
            t1.start();
            train_error = network.get_loss_error(train_batch_size);   
            train_accuracy = network.get_accuracy(train_batch_size);
            network.validataion_setting(validation_batch_size);
            network.forward_propagation(validation_batch_size);
            validation_error = network.get_loss_error(validation_batch_size);   
            validation_accuracy = network.get_accuracy(validation_batch_size);
            printf("train_e: %1.6f, train_acc: %1.6f, val_e: %1.6f, val_acc: %1.6f\n",
                    train_error,train_accuracy,validation_error,validation_accuracy);
            device2host_time += t1.finish();
        }
    } 
    pthread_cancel(thread_handles); // thread sync

    total_time = t3.finish();


    network.test_setting(test_batch_size);
    network.forward_propagation(test_batch_size);
    test_error = network.get_loss_error(test_batch_size);   
    test_accuracy = network.get_accuracy(test_batch_size);
    printf("test error : %1.6f, test acc : %1.6f\n",test_error,test_accuracy);

    cout<<"setting_time : "<<setting_time<<endl;
    cout<<"read_time  : "<<read_time/max_iteration<<endl;
    cout<<"host to device time : "<<host2device_time/max_iteration<<endl;
    cout<<"GPU time : "<<GPU_time/max_iteration<<endl;
    cout<<"device to host time : "<<device2host_time/max_iteration*print_per_iteration<<endl;
    cout<<"total time : "<<total_time<<endl;

    return 0;
}

void* thread_function(void *id)
{
    while(1)
    {
        pthread_mutex_lock(&mutex);
        if(sync_stat == true)
        {
            t2.start(); // start 
            sync_stat = false;
            data.third_read_train(idx);
            read_time +=  t2.finish(); // finish 
        }
        pthread_mutex_unlock(&mutex);
    }
}


int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch(devProp.major)
    {
        case 2 ://Fermi
            if (devProp.major == 1) cores = mp*48;
            else cores = mp*32;
            break;
        case 3 : //Kepler
            cores = mp*192;
            break;
        case 5 : //Maxwell
            cores = mp*128;
            break;
        case 6 : //Pascal
            if(devProp.minor == 1) cores = mp *128;
            else if(devProp.minor == 0) cores = mp*64;
            else printf("Unknown device type\n");
            break;
        defualt :
            printf("Unknown device type\n");
            break;

    }
    return cores;
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
        cout<<"clock rate : "<<deviceProperties.clockRate/1048576.0<<" GHz"<<endl;
        cout<<"cores : "<<getSPcores(deviceProperties)<<endl;
        cout<<"totalGlobalMem : "<<deviceProperties.totalGlobalMem/1073741824.0<<" GByte"<<endl;

    }
    cout<<endl;
    if(n > devicesCount && n < 0) return false;
    else
    {
        cudaSetDevice(n);

        return true;
    }
}

