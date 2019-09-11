#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "sub_main.h"
#include "my_mnist_class.h"
#include "my_graph_net.cuh"

using namespace std;


My_Timer t1,t2,t3;

double setting_time =0.0;
double hd2host_time =0.0;
double host2device_time =0.0;
double gpu_time =0.0;
double device2host_time =0.0;
double total_time =0.0;




int main(int argc, char **argv)
{
    t1.start();


    if(argc != 2)
    {
        printf("USAGE : %s [gpu num]\n",argv[0]);
        exit(1);
    }

    int gpu_num = atoi(argv[1]);
    ChoseGpuAvailable(gpu_num);
    srand((unsigned)time(NULL));

    //------
    t2.start();
    
    MY_GRAPH_NET net;
    MY_PARA_MANAGER para;
    MY_ADAM_OPTIMIZER adam;
    MY_REGULARIZATION maxnorm;

    MY_MATRIX_DEVICE *u1,*ta1,*noise, *r1,*r1_drop, *r2,*r2_drop, *r3,*r3_drop, *r4,*r4_drop, *r5,*r6;
    MY_MATRIX_DEVICE *w1, *w2, *b1, *b2, *w3, *b3, *w4, *b4;

    int mini_batch_size = 100;
    int n1 = 784;
    int n2 = 2048;
    int n3 = 2048;
    int n4 = 2048;
    int n5 = 10;

    u1 = para.set("u1",n1,mini_batch_size);

    w1 = para.set("w1",n2,n1);
    b1 = para.set("b1",n2);
    w2 = para.set("w2",n3,n2);
    b2 = para.set("b2",n3);
    w3 = para.set("w3",n4,n3);
    b3 = para.set("b3",n4);
    w4 = para.set("w4",n5,n4);
    b4 = para.set("b4",n5);

    ta1 = para.set("ta1",n5,mini_batch_size);
    float mean = 0.0;
    float std = 0.01;
    my_set_gaussian(mean,std,w1,w2,w3,w4,NULL);


  //  noise = net.white_noise(n1,mini_batch_size,0.01);

    //r1 = net.min(net.adding_point(u1,noise),1);

    r1_drop = net.inverted_dropout(u1,0.8);

    r2 = net.relu(net.add_bias(net.multi(w1,r1_drop),b1));

    r2_drop = net.inverted_dropout(r2,0.5);
    r3 = net.relu(net.add_bias(net.multi(w2,r2_drop),b2));

    r3_drop = net.inverted_dropout(r3,0.5);
    r4 = net.relu(net.add_bias(net.multi(w3,r3_drop),b3));

    r4_drop = net.inverted_dropout(r4,0.5);
    r5 = net.sigmoid(net.add_bias(net.multi(w4,r4_drop),b4));

    r6 = net.binary_cross_entropy(r5,ta1);



    net.network_init(rand());
    adam.set_hyperpara(0.0001);

    adam.set_para(w1,w2,w3,w4,b1,b2,b3,b4,NULL);
//    maxnorm.set_para(MAX_NORM,2.0,w1,w2,w3,w4,NULL);
  
    EXAMPLE_MNIST_CLASS data;
    data.first_read("../../MNIST_data");
    data.second_init(mini_batch_size);  

    make_index(60000 - mini_batch_size);
    setting_time = t2.finish(); 

    for(int i = 0 ; i < 10000 ; i++){

        t2.start();
        data.third_read_train(get_next());
        hd2host_time = t2.finish();    

        t2.start();
        my_host2device(data.cur_input,u1->x,n1*mini_batch_size);
        my_host2device(data.cur_target,ta1->x,n5*mini_batch_size);
        host2device_time = t2.finish();

       
        t2.start();

        net.foreward();
        net.backward();
        adam.update();
        //maxnorm.update();

        gpu_time = t2.finish();


        if(i % 1000 == 0){

            t2.start();
            printf("train err : %f, train acc : %f\n",net.average_absolute(r6),net.accuracy(r5,ta1));
            shuffle_index();
            device2host_time = t2.finish();
        }



    }
    //-----------

    //    my_print(r1);
    /*    my_print(r2);
          my_print(r3);


          my_print(r4);

          my_print(r5);
          my_print(r6);
          my_print(a);
          my_print(w1);
          my_print(u1);
          */ 


    total_time = t1.finish();
    cout<<endl;
    cout<<"setting time : "<<setting_time<<endl;
    cout<<"hd2host time : "<<hd2host_time<<endl;
    cout<<"host2device time : "<<host2device_time<<endl;
    cout<<"gpu time : "<<gpu_time<<endl;
    cout<<"device2host time : "<<device2host_time<<endl;
    cout<<"total time : "<<total_time<<endl;


    my_para_write("para.txt",w1,w2,w3,w4,b1,b2,b3,b4,NULL);
    
    return 0;
}
