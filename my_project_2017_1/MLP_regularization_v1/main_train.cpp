#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <pthread.h>
#include <time.h>
#include "MLP_basic.h"
#include "my_mnist_class.h"
#include "sub_fucntion.h"

using namespace std;

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


EXAMPLE_MNIST_CLASS *data;
MLP_basic *network;


long neurals[5] = {784,1024,1024,2048,10};
long layers = 5;


float alpha = 0.0001;       //learning rate
float ramda = 0.0;          //weight decay para
float beta1 = 0.9;          //adam para
float beta2 = 0.999;        //adam paea
float dropout_rate[5] = {0.8,0.5,0.5,0.5,1.0};
float maxnorm_constraints = 2.0;


long train_batch_size = 256;
long validation_batch_size = 5000;
long test_batch_size = 10000;
long maximum_train_set = 55000;


unsigned long max_iteration = 1000000;
unsigned long print_per_iteration = 10000;


long idx = 0;
int gpu_set = 0;
char *name_txt;


int main(int argc,char **argv)
{
    t3.start();
   

    if(argc != 6)
    {
        printf("USAGE : %s [learning rate] [batch size] [maxnorm] [gpu_id] [result.txt]\n",argv[0]);
        exit(1);
    }
   
    alpha = atof(argv[1]);
    train_batch_size = atoi(argv[2]);
    maxnorm_constraints = atof(argv[3]);
    gpu_set = atoi(argv[4]);
    name_txt = argv[5];
    FILE *fd;
    fd = fopen(name_txt,"a");
    if(fd == NULL) 
    {
        printf("file open fail\n");
        exit(0);
    }



    ChoseGpuAvailable(gpu_set);
    pthread_mutex_init(&mutex,NULL);
    srand(time(NULL));
    data = new EXAMPLE_MNIST_CLASS;
    network = new MLP_basic;



    t1.start(); // start setting

    data->first_read("../../MNIST_data");
    data->second_init(train_batch_size);  
    data->third_read_train(0);
    
    long max_batch_size;
    max_batch_size = (validation_batch_size > test_batch_size)? validation_batch_size : test_batch_size;
    max_batch_size = (max_batch_size > train_batch_size)? max_batch_size : train_batch_size;


    network->seed(rand()); 
    network->init(neurals,layers,max_batch_size,alpha,ramda,beta1,beta2);
    network->first_random_parameter();
    network->second_validation_test_set_host_device(data->validation_images, data->validation_labels, 
                validation_batch_size, data->test_images, data->test_labels, test_batch_size);
    setting_time =  t1.finish(); // finish setting
 
       
    pthread_create(&thread_handles,NULL,thread_function,(void*)NULL);
    float *temp_input, *temp_target;
    
    make_index(maximum_train_set + validation_batch_size - train_batch_size);

    for(long iter = 1 ; iter <= max_iteration; iter++){
        
        pthread_mutex_lock(&mutex);

        idx = get_next();
        sync_stat = true;
        temp_input = data->cur_input;
        temp_target = data->cur_target;

        pthread_mutex_unlock(&mutex);


        t1.start();// timer start
        network->third_train_set_host_device(temp_input, temp_target,train_batch_size); 
        host2device_time += t1.finish(); //timer finish

        t1.start(); //timer start
        network->train_forward_propagation(train_batch_size,dropout_rate);
        network->delta_rule(train_batch_size);
        network->update_adam(train_batch_size,maxnorm_constraints);

        GPU_time += t1.finish(); //timer finish

        if(iter% print_per_iteration == 0)
        {
            t1.start();
            train_error = network->get_loss_error(train_batch_size);   
            train_accuracy = network->get_accuracy(train_batch_size);
            network->test_setting(test_batch_size);
            network->test_forward_propagation(test_batch_size);
            test_error = network->get_loss_error(test_batch_size);   
            test_accuracy = network->get_accuracy(test_batch_size);


            printf("train_e: %1.6f, train_acc: %1.6f, test_e: %1.6f, test_acc: %1.6f\n",
                    train_error,train_accuracy,test_error,test_accuracy);
            device2host_time += t1.finish();

            network->seed(rand());
            shuffle_index();

        }
    } 
    pthread_cancel(thread_handles); // thread sync

    total_time = t3.finish();


    network->test_setting(test_batch_size);
    network->test_forward_propagation(test_batch_size);
    test_error = network->get_loss_error(test_batch_size);   
    test_accuracy = network->get_accuracy(test_batch_size);
    
    for(int i = 0 ; i < layers -1 ;i++) fprintf(fd,"%ld-",neurals[i]); 
    fprintf(fd,"%ld, ",neurals[layers-1]);
    for(int i = 0 ; i < layers -1 ;i++) fprintf(fd,"%1.1f-",dropout_rate[i]); 
    fprintf(fd,"%1.1f, ",dropout_rate[layers-1]);
    
    fprintf(fd,"%1.8f, %3ld, %2.2f, %2.6f, %1.4f\n",alpha,train_batch_size,maxnorm_constraints,test_error,test_accuracy);


    fclose(fd);
    delete data;
    delete network;

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
            data->third_read_train(idx);
            read_time +=  t2.finish(); // finish 
        }
        pthread_mutex_unlock(&mutex);
    }
}


