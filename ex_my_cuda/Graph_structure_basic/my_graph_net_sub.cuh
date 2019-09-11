#ifndef __MY_GRAPH_NET_SUB_CU__
#define __MY_GRAPH_NET_SUB_CU__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include <opencv2/opencv.hpp>

using namespace std;

#define CUDA_CALL(x)          if((x) != cudaSuccess){\
    printf("CUDA Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}

#define CUBLAS_CALL(x)        if((x) != CUBLAS_STATUS_SUCCESS){\
    printf("CUBLAS Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}

#define CURAND_CALL(x)        if((x) != CURAND_STATUS_SUCCESS){\
    printf("CURAND Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}

#define MY_FUNC_ERROR(x)          if(!(x)){\
    printf("my func Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}


#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> column : x, i -> row : y, column major


class MY_MATRIX_DEVICE{
    public :
        float *x;
        float *grad_x;
        int row, column;
        float rate;
        char name[64];
        
        MY_MATRIX_DEVICE();
        ~MY_MATRIX_DEVICE();
};

struct _node_matrix{

    MY_MATRIX_DEVICE *data;

    struct _node_matrix *next;
    struct _node_matrix *prev;
};

class MY_MATRIX_DEQUE{

    public : 
        _node_matrix *head;
        _node_matrix *tail;
 
        MY_MATRIX_DEQUE();
        ~MY_MATRIX_DEQUE();
        
        bool IsEmpty();
        
        void AddFirst(MY_MATRIX_DEVICE *pdata);
        void AddLast(MY_MATRIX_DEVICE *pdata);
        void RemoveFirst();
        void RemoveLast();

};


enum GATE_STAT{
    FOREWARD = 0,
    BACKWARD,
    TEST

};


struct _node_graph_net{

    MY_MATRIX_DEVICE *in1;
    MY_MATRIX_DEVICE *in2;
    MY_MATRIX_DEVICE *out;
    void (*operate)(void* ,MY_MATRIX_DEVICE* , MY_MATRIX_DEVICE*, MY_MATRIX_DEVICE*,GATE_STAT);
    struct _node_graph_net *next;
    struct _node_graph_net *prev;
};

class MY_GRAPH_NET_DEQUE{

    public : 
        _node_graph_net *head;
        _node_graph_net *tail;
 
        MY_GRAPH_NET_DEQUE();
        ~MY_GRAPH_NET_DEQUE();
        
        bool IsEmpty();
        
        void AddFirst(MY_MATRIX_DEVICE*,MY_MATRIX_DEVICE*,MY_MATRIX_DEVICE*,
                void (*func)(void* , MY_MATRIX_DEVICE*, MY_MATRIX_DEVICE*,MY_MATRIX_DEVICE*, GATE_STAT));
        void AddLast(MY_MATRIX_DEVICE *, MY_MATRIX_DEVICE *, MY_MATRIX_DEVICE *,
                void (*func)(void* , MY_MATRIX_DEVICE*, MY_MATRIX_DEVICE*,MY_MATRIX_DEVICE*, GATE_STAT));
        void RemoveFirst();
        void RemoveLast();

};


class MY_PARA_MANAGER{
    private :
        MY_MATRIX_DEQUE deque_matrix;


    public : 
        MY_PARA_MANAGER();
        ~MY_PARA_MANAGER();

        MY_MATRIX_DEVICE* set(const char *name, int row, int column = 1);


};

void my_set_gaussian(float mean, float std, ...);

void my_para_write(const char *filename, ...);
void my_para_read(const char *filename, ...);

void my_host2device(float *host, float* device, int n);
void my_print(MY_MATRIX_DEVICE *pa);


void get_mnist_image(const char *name, MY_MATRIX_DEVICE *pa);


#endif
