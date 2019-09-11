#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "my_mnist_class.h"

using namespace std;

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



int main(int argc, char **argv)
{

    int batch_size = 128;
    int index;

    double setting_time;
    double thread_time;

    My_Timer t;
    EXAMPLE_MNIST_CLASS a;


    t.start();

    a.first_read("../../MNIST_data");
    a.second_init(batch_size);  

    setting_time =  t.finish();


    t.start();

    index = 0;
    a.third_read_train(index);

    thread_time =  t.finish();



    cout<<"setting_time : "<<setting_time<<endl;
    cout<<"thread_time  : "<<thread_time<<endl;
    return 0;
}
