#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "sub_main.h"
#include "my_mnist_class.h"
#include "my_graph_net.cuh"
#include "my_graph_net_sub.cuh"

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
    
    MY_GRAPH_NET net,pre_process;

    MY_PARA_MANAGER para;
    MY_ADAM_OPTIMIZER adam;
    MY_REGULARIZATION maxnorm;

    MY_MATRIX_DEVICE *u1,*u2,*noise, *r1,*r2,*r3,*r4,*r5,*r6,*r7,*r8,*r9,*r10,
                     *r11,*r12,*r13,*r14,*r15,*r16,*r17,*r18,*r19,*r20,
                     *r21,*r22,*r23,*r24,*r25;
    MY_MATRIX_DEVICE *w1, *w2, *w3, *w4, *w5, *w6, *w7,*b1, *b2, *b3, *b4, *b5, *b6, *b7;

    int mini_batch_size = 2;
    int n1 = 1568;
    int n2 = 2048;
    int n3 = 2048;
    int n4 = 2048;
    int n5 = 2048;
    int n6 = 2048;
    int n7 = 1568;
    int n8 = 784;



    u1 = para.set("u1",n1/2,mini_batch_size);

    u2 = para.set("u2",n1/2,mini_batch_size);


    w1 = para.set("w1",n2,n1);
    b1 = para.set("b1",n2);
    w2 = para.set("w2",n3,n2);
    b2 = para.set("b2",n3);
    w3 = para.set("w3",n4,n3);
    b3 = para.set("b3",n4);
    w4 = para.set("w4",n5,n4);
    b4 = para.set("b4",n5);
    w5 = para.set("w5",n6,n5);
    b5 = para.set("b5",n6);
    w6 = para.set("w6",n7,n6);
    b6 = para.set("b6",n7);
    w7 = para.set("w7",n8,n7);
    b7 = para.set("b7",n8);




    float mean = 0.0;
    float std = 0.01;
    my_set_gaussian(mean,std,w1,w2,w3,w4,w5,w6,w7,NULL);

//    my_para_read("para.txt",w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7,NULL);
    MY_MATRIX_DEVICE *x1,*x2,*x3,*x4,*x5;

    pre_process.dividing_point(u2,&x1,&x2);
   // x3 = pre_process.adding_point(u1,x2);
   // x4 = pre_process.min(x3,1.0);
    
    x4 = pre_process.white_noise(n1/2,mini_batch_size,0.8);
 
    x1 = pre_process.white_noise(n1/2,mini_batch_size,0.8);



    
    x2 = pre_process.merge(x4,x1);
    x5 = pre_process.rand_scale(x2,0.0,1.0);
    r5 = pre_process.elu(pre_process.add_bias(pre_process.multi(w1,x5),b1));


/*   
    net.dividing_point(x5,&r3,&r4);
    r5 = net.elu(net.add_bias(net.multi(w1,r3),b1));

    net.dividing_point(r5,&r6,&r7);
    r8 = net.elu(net.add_bias(net.multi(w2,r6),b2));
    r9 = net.adding_point(r8,r7);

    net.dividing_point(r9,&r10,&r11);
    r12 = net.elu(net.add_bias(net.multi(w3,r10),b3));
    r13 = net.adding_point(r12,r11);

    net.dividing_point(r13,&r14,&r15);
    r16 = net.elu(net.add_bias(net.multi(w4,r14),b4));
    r17 = net.adding_point(r16,r15);

    net.dividing_point(r17,&r18,&r19);
    r20 = net.elu(net.add_bias(net.multi(w5,r18),b5));
    r21 = net.adding_point(r19,r18);

    r22 = net.elu(net.add_bias(net.multi(w6,r21),b6));
    r23 = net.adding_point(r22,r4);

    r24 = net.sigmoid(net.add_bias(net.multi(w7,r23),b7));

   
    r25 = net.binary_cross_entropy(r24,u1);
*/


    pre_process.network_init();
   // net.network_init(rand());
    
    adam.set_hyperpara(0.000001);

    adam.set_para(w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7,NULL);
    maxnorm.set_para(MAX_NORM,3.0,w1,w2,w3,w4,w5,w6,w7,NULL);
  
    EXAMPLE_MNIST_CLASS data;
    data.first_read("../../MNIST_data");
    data.second_init(mini_batch_size);  

    make_index(60000 - mini_batch_size);
    setting_time = t2.finish(); 
   
    for(int i = 0 ; i < 1 ; i++){
/*        

        t2.start();
        data.third_read_train(get_next());
        hd2host_time = t2.finish();    

        t2.start();
        my_host2device(data.cur_input,u1->x,n1/2*mini_batch_size);
        host2device_time = t2.finish();
        
        t2.start();
        data.third_read_train(get_next());
        hd2host_time = t2.finish();    

        t2.start();
        my_host2device(data.cur_input,u2->x,n1/2*mini_batch_size);
        host2device_time = t2.finish();
*/

       
        t2.start();

        pre_process.foreward();

        pre_process.backward();
        
//        net.foreward();
//        net.backward();
//        adam.update();
//        maxnorm.update();

        gpu_time = t2.finish();


        if(i % 10000 == 0){

            t2.start();
            printf("train err : %f\n",net.average_absolute(r25));
            shuffle_index();
            device2host_time = t2.finish();
        }



    }
    //-----------
   
    //get_mnist_image("resNet_input", x4);
 

    //get_mnist_image("resNet_ref_input", u2);
 
    //get_mnist_image("resNet_output", r24);

 my_print(x5);
 my_print(x2);



    total_time = t1.finish();
    cout<<endl;
    cout<<"setting time : "<<setting_time<<endl;
    cout<<"hd2host time : "<<hd2host_time<<endl;
    cout<<"host2device time : "<<host2device_time<<endl;
    cout<<"gpu time : "<<gpu_time<<endl;
    cout<<"device2host time : "<<device2host_time<<endl;
    cout<<"total time : "<<total_time<<endl;


    //my_para_write("para.txt",w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7,NULL);
 
    return 0;
}
