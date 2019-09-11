#include "my_graph_net_sub.cuh"
//#include "my_graph_net.cuh"
//#include "my_device_func.cuh"
//---------------------------------------
MY_MATRIX_DEVICE :: MY_MATRIX_DEVICE()
{
    x = NULL;
    grad_x = NULL;
    row=0;
    column =0;
    rate = 0.0;
              

}

MY_MATRIX_DEVICE :: ~MY_MATRIX_DEVICE()
{
    if(x != NULL) CUDA_CALL(cudaFree(x));
    if(grad_x != NULL) CUDA_CALL(cudaFree(grad_x));

}



MY_MATRIX_DEQUE :: MY_MATRIX_DEQUE()
{
    head = NULL;
    tail = NULL;

}
MY_MATRIX_DEQUE :: ~MY_MATRIX_DEQUE()
{
    if(head != NULL)
    {
        while(IsEmpty() == false){
            RemoveLast();
        }
    }

}
bool MY_MATRIX_DEQUE :: IsEmpty()
{
    if(head == NULL) return true;
    else return false;

}

void MY_MATRIX_DEQUE :: AddFirst(MY_MATRIX_DEVICE *pdata)
{
    _node_matrix *newNode = (_node_matrix*)malloc(sizeof(_node_matrix));

    newNode->data = pdata;

    newNode->next = head;
    if(IsEmpty()) tail = newNode;
    else head->prev = newNode;

    newNode->prev = NULL;
    head = newNode;

}
void MY_MATRIX_DEQUE :: AddLast(MY_MATRIX_DEVICE *pdata)
{
   _node_matrix *newNode = (_node_matrix*)malloc(sizeof(_node_matrix));

    newNode->data = pdata;

    newNode->prev = tail;
    if(IsEmpty()) head = newNode;
    else tail->next = newNode;

    newNode->next = NULL;
    tail = newNode;


}
void MY_MATRIX_DEQUE :: RemoveFirst()
{
    _node_matrix *rnode = head;

    delete rnode->data;//

    head = head->next;
    free(rnode);
    if(head == NULL) tail = NULL;
    else head->prev = NULL;

}
void MY_MATRIX_DEQUE :: RemoveLast()
{
    _node_matrix *rnode = tail;

    delete rnode->data;//

    tail = tail->prev;
    free(rnode);
    if(tail == NULL) head = NULL;
    else tail->next = NULL;


}

//---------------------------------------
MY_GRAPH_NET_DEQUE :: MY_GRAPH_NET_DEQUE()
{
    head = NULL;
    tail = NULL;
}
MY_GRAPH_NET_DEQUE :: ~MY_GRAPH_NET_DEQUE()
{
    if(head != NULL)
    {
        while(IsEmpty() == false) RemoveLast();
    }
}
bool MY_GRAPH_NET_DEQUE :: IsEmpty()
{
    if(head == NULL) return true;
    else return false;
}

void MY_GRAPH_NET_DEQUE :: AddFirst(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc,
        void (*func)(void*, MY_MATRIX_DEVICE*, MY_MATRIX_DEVICE*, MY_MATRIX_DEVICE*, GATE_STAT))
{
    _node_graph_net *newNode = (_node_graph_net*)malloc(sizeof(_node_graph_net));

    newNode->in1 = pa;
    newNode->in2 = pb;
    newNode->out = pc;
    newNode->operate = func;

    newNode->next = head;
    if(IsEmpty()) tail = newNode;
    else head->prev = newNode;

    newNode->prev = NULL;
    head = newNode;
}
void MY_GRAPH_NET_DEQUE :: AddLast(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc,
        void (*func)(void*, MY_MATRIX_DEVICE*, MY_MATRIX_DEVICE*, MY_MATRIX_DEVICE*, GATE_STAT))
{
    _node_graph_net *newNode = (_node_graph_net*)malloc(sizeof(_node_graph_net));

    newNode->in1 = pa;
    newNode->in2 = pb;
    newNode->out = pc;
    newNode->operate = func;

    newNode->prev = tail;
    if(IsEmpty()) head = newNode;
    else tail->next = newNode;

    newNode->next = NULL;
    tail = newNode;

}
void MY_GRAPH_NET_DEQUE :: RemoveFirst()
{
    _node_graph_net *rnode = head;

    //

    head = head->next;
    free(rnode);
    if(head == NULL) tail = NULL;
    else head->prev = NULL;
}
void MY_GRAPH_NET_DEQUE :: RemoveLast()
{
    _node_graph_net *rnode = tail;

    //

    tail = tail->prev;
    free(rnode);
    if(tail == NULL) head = NULL;
    else tail->next = NULL;

}

