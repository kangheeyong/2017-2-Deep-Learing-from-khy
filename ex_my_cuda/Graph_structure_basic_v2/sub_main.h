#ifndef __SUB_MAIN_CPP__
#define __SUB_MAIN_CPP__

#include <time.h>

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



bool ChoseGpuAvailable(int n);


void make_index(int max_size);
void shuffle_index();

int get_index(int n);
int get_next();







#endif
