#include "my_graph_net.cuh"
#include "my_device_func.cuh"
#include <stdarg.h>


MY_GRAPH_NET ::  MY_GRAPH_NET()
{
    CURAND_CALL(curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT));
    CUBLAS_CALL(cublasCreate(&handle));


}
MY_GRAPH_NET :: ~MY_GRAPH_NET()
{
    CUBLAS_CALL(cublasDestroy(handle));
    CURAND_CALL(curandDestroyGenerator(rand_gen));


}

void MY_GRAPH_NET :: foreward()
{
    _node_graph_net *cur = deque_operate.head;
    do{
        cur->operate(this,cur->in1,cur->in2,cur->out,FOREWARD);
        cur = cur->next;
    }
    while(cur != NULL);

}

void MY_GRAPH_NET :: test()
{
    _node_graph_net *cur = deque_operate.head;
    do{
        cur->operate(this,cur->in1,cur->in2,cur->out,TEST);
        cur = cur->next;
    }
    while(cur != NULL);

}




void MY_GRAPH_NET :: backward()
{
    _node_graph_net *cur = deque_operate.tail;
    do{
        cur->operate(this,cur->in1,cur->in2,cur->out,BACKWARD);
        cur = cur->prev;
    }
    while(cur != NULL);


}

void MY_GRAPH_NET :: network_init(int seed)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    int max_row = 0;
    int max_column = 0;

    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_gen,seed));
 
    _node_graph_net *cur = deque_operate.tail;

    do{

        if(cur->in1 != NULL)
        {
            if(cur->in1->row > max_row) max_row = cur->in1->row;
            if(cur->in1->column > max_column) max_column = cur->in1->column;
            if(((cur->in1->row) > 0) && ((cur->in1->column) > 0) && (cur->in1->x) == NULL)  
            {
                CUDA_CALL(cudaMalloc(&(cur->in1->grad_x),sizeof(float)*(cur->in1->row)*(cur->in1->column)));
                CUDA_CALL(cudaMalloc(&(cur->in1->x),sizeof(float)*(cur->in1->row)*(cur->in1->column)));
                blocksPerGride = ((cur->in1->row)*(cur->in1->column) + threadsPerBolck -1)/threadsPerBolck; 
                make_ones<<<blocksPerGride, threadsPerBolck>>>(cur->in1->grad_x,(cur->in1->row)*(cur->in1->column));  
            }
        }

        if(cur->in2 != NULL)
        {
            if(cur->in2->row > max_row) max_row = cur->in2->row;
            if(cur->in2->column > max_column) max_column = cur->in2->column;
            if(((cur->in2->row) > 0) && ((cur->in2->column) > 0) && (cur->in2->x) == NULL ) 
            {
                CUDA_CALL(cudaMalloc(&(cur->in2->grad_x),sizeof(float)*(cur->in2->row)*(cur->in2->column)));
                CUDA_CALL(cudaMalloc(&(cur->in2->x),sizeof(float)*(cur->in2->row)*(cur->in2->column)));
                blocksPerGride = ((cur->in2->row)*(cur->in2->column) + threadsPerBolck -1)/threadsPerBolck; 
                make_ones<<<blocksPerGride, threadsPerBolck>>>(cur->in2->grad_x,(cur->in2->row)*(cur->in2->column));  
            }
        }

        if(cur->out != NULL)
        {
            if(cur->out->row > max_row) max_row = cur->out->row;
            if(cur->out->column > max_column) max_column = cur->out->column;
            if(((cur->out->row) > 0) && ((cur->out->column) > 0) && (cur->out->x) == NULL ) 
            {
                CUDA_CALL(cudaMalloc(&(cur->out->grad_x),sizeof(float)*(cur->out->row)*(cur->out->column)));
                CUDA_CALL(cudaMalloc(&(cur->out->x),sizeof(float)*(cur->out->row)*(cur->out->column)));
                blocksPerGride = ((cur->out->row)*(cur->out->column) + threadsPerBolck -1)/threadsPerBolck; 
                make_ones<<<blocksPerGride, threadsPerBolck>>>(cur->out->grad_x,(cur->out->row)*(cur->out->column));  
            }
        }
           
        cur = cur->prev;

    }
    while(cur != NULL);

    CUDA_CALL(cudaMalloc(&(d_ones.x),sizeof(float)*max_row*max_column));
    d_ones.row = max_row;
    d_ones.column = max_column;
    blocksPerGride = ((d_ones.row)*(d_ones.column) + threadsPerBolck -1)/threadsPerBolck; 
    make_ones<<<blocksPerGride, threadsPerBolck>>>(d_ones.x,(d_ones.row)*(d_ones.column));  


    CUDA_CALL(cudaMalloc(&(d_temp.x),sizeof(float)*max_row*max_column));
    d_temp.row = max_row;
    d_temp.column = max_column;


}


void gate_multi(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    float const one = 1.0;
    float const zero = 0.0;
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
   

    if((stat == FOREWARD) || (stat == TEST))
    {
        CUBLAS_CALL(cublasSgemm(((MY_GRAPH_NET*)pthis)->handle, CUBLAS_OP_N, CUBLAS_OP_N, pc->row, pc->column, pb->row,
                    &one, pa->x, pa->row, pb->x, pb->row,  &zero, pc->x, pc->row));
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        transpose<<<blocksPerGride, threadsPerBolck>>>(((MY_GRAPH_NET*)pthis)->d_temp.x, pa->x,
                pa->row, pa->column, (pa->row)*(pa->column));  
    
        CUBLAS_CALL(cublasSgemm(((MY_GRAPH_NET*)pthis)->handle, CUBLAS_OP_N, CUBLAS_OP_N, pb->row, pb->column, pc->row,
                    &one, ((MY_GRAPH_NET*)pthis)->d_temp.x, pb->row, pc->grad_x, pc->row,  &zero, pb->grad_x, pb->row));


        blocksPerGride = (pb->row*pb->column + threadsPerBolck -1)/threadsPerBolck;
        transpose<<<blocksPerGride, threadsPerBolck>>>(((MY_GRAPH_NET*)pthis)->d_temp.x, pb->x,
                pb->row, pb->column, (pb->row)*(pb->column));  
    
        CUBLAS_CALL(cublasSgemm(((MY_GRAPH_NET*)pthis)->handle, CUBLAS_OP_N, CUBLAS_OP_N, pa->row, pa->column, pc->column,
                    &one, pc->grad_x, pc->row, ((MY_GRAPH_NET*)pthis)->d_temp.x, pc->column, &zero, pa->grad_x, pa->row));
    }
}



MY_MATRIX_DEVICE* MY_GRAPH_NET :: multi(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb)
{
    MY_FUNC_ERROR(pa->column == pb->row);

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pb->column;

    deque_operate.AddLast(pa,pb,pc,&gate_multi);
    return pc;
}


void gate_bias_add(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    float const one = 1.0;
    float const zero = 0.0;
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        add_bias<<<blocksPerGride, threadsPerBolck>>>(pa->x,pb->x,pc->x,pa->row,(pc->row)*(pc->column));  
    }
    else if(stat ==BACKWARD)
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,(pc->row)*(pc->column));  
        CUBLAS_CALL(cublasSgemm(((MY_GRAPH_NET*)pthis)->handle, CUBLAS_OP_N, CUBLAS_OP_N, pb->row, pb->column, pc->column,
                    &one, pc->grad_x, pc->row, ((MY_GRAPH_NET*)pthis)->d_ones.x, pc->column,  &zero, pb->grad_x, pb->row));
    }
}



MY_MATRIX_DEVICE* MY_GRAPH_NET :: add_bias(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb)
{
   
    MY_FUNC_ERROR((pa->row == pb->row)&&(pb->column == 1));
    
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,pb,pc,&gate_bias_add);
    return pc;
}


void gate_relu(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        relu<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,(pa->row)*(pa->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        relu_inv<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pa->x,(pa->row)*(pa->column));  
      
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        scalar_multi<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,pa->grad_x,(pc->row)*(pc->column));  
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: relu(MY_MATRIX_DEVICE *pa)
{
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,NULL,pc,&gate_relu);
    return pc;
}


void gate_elu(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        elu<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,(pa->row)*(pa->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        elu_inv<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pa->x,(pa->row)*(pa->column));  
      
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        scalar_multi<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,pa->grad_x,(pc->row)*(pc->column));  
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: elu(MY_MATRIX_DEVICE *pa)
{
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,NULL,pc,&gate_elu);
    return pc;
}




void gate_binary_cross_entropy(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD))
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        binary_cross_entropy<<<blocksPerGride, threadsPerBolck>>>(pa->x,pb->x,pc->x,(pc->row)*(pc->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        binary_cross_entropy_inv<<<blocksPerGride, threadsPerBolck>>>(pa->x,pb->x,pa->grad_x,(pa->row)*(pa->column));  
      
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        scalar_multi<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,pa->grad_x,(pc->row)*(pc->column));  
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: binary_cross_entropy(MY_MATRIX_DEVICE *pa,MY_MATRIX_DEVICE *pb)
{
    
    MY_FUNC_ERROR((pa->row == pb->row)&&(pa->column == pb->column));
    
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,pb,pc,&gate_binary_cross_entropy);
    return pc;

}



void gate_least_squares(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD))
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        least_squares<<<blocksPerGride, threadsPerBolck>>>(pa->x,pb->x,pc->x,(pc->row)*(pc->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        least_squares_inv<<<blocksPerGride, threadsPerBolck>>>(pa->x,pb->x,pa->grad_x,(pa->row)*(pa->column));  
      
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        scalar_multi<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,pa->grad_x,(pc->row)*(pc->column));  
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: least_squares(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb)
{
    MY_FUNC_ERROR((pa->row == pb->row)&&(pa->column == pb->column));
    
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,pb,pc,&gate_least_squares);
    return pc;

}


void gate_inverted_dropout(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    float const dropout_rate = pb->rate;

    if(stat == FOREWARD)
    {
        CURAND_CALL(curandGenerateUniform(((MY_GRAPH_NET*)pthis)->rand_gen,pb->x,(pb->row)*(pb->column)));
     
        blocksPerGride = (pb->row*pb->column + threadsPerBolck -1)/threadsPerBolck;
        dropout_table<<<blocksPerGride, threadsPerBolck>>>(pb->x,dropout_rate,(pb->row)*(pb->column));  
    
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        dropout<<<blocksPerGride, threadsPerBolck>>>(pa->x,pb->x,pc->x,dropout_rate,(pc->row)*(pc->column));  
    }
    else if(stat == TEST)
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pa->x,pc->x,(pc->row)*(pc->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        dropout<<<blocksPerGride, threadsPerBolck>>>(pc->grad_x,pb->x,pa->grad_x,dropout_rate,(pc->row)*(pc->column));  
  
    }
}



MY_MATRIX_DEVICE* MY_GRAPH_NET :: inverted_dropout(MY_MATRIX_DEVICE *pa, float rate)
{

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);
    
    MY_MATRIX_DEVICE *pb = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pb);
        
    pc->row = pa->row;
    pc->column = pa->column;
 
    pb->rate = rate;
    pb->row = pa->row;
    pb->column = pa->column;

    deque_operate.AddLast(pa,pb,pc,&gate_inverted_dropout);
    return pc;


}

void gate_sigmoid(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, 
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        sigmoid<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,(pa->row)*(pa->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        sigmoid_inv<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pa->x,(pa->row)*(pa->column));  
      
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        scalar_multi<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,pa->grad_x,(pc->row)*(pc->column));  
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: sigmoid(MY_MATRIX_DEVICE *pa)
{
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,NULL,pc,&gate_sigmoid);
    return pc;

}


void gate_tanh(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        tanh<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,(pa->row)*(pa->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        tanh_inv<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pa->x,(pa->row)*(pa->column));  
      
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        scalar_multi<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,pa->grad_x,(pc->row)*(pc->column));  
    }
}



MY_MATRIX_DEVICE* MY_GRAPH_NET :: tanh(MY_MATRIX_DEVICE *pa)
{
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,NULL,pc,&gate_tanh);
    return pc;


}

void gate_adding_point(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        add<<<blocksPerGride, threadsPerBolck>>>(pa->x,pb->x,pc->x,(pc->row)*(pc->column));  
    }
    else if(stat ==BACKWARD)
    {
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,(pc->row)*(pc->column));  
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pb->grad_x,pc->grad_x,(pc->row)*(pc->column));  
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: adding_point(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb)
{
    MY_FUNC_ERROR(((pa->column == pb->column) && (pa->row == pb->row)));

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,pb,pc,&gate_adding_point);
    return pc;

}

void gate_dividing_point(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pb->x,pa->x,(pa->row)*(pa->column));  
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,(pa->row)*(pa->column));  
    }
    else if(stat ==BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        add<<<blocksPerGride, threadsPerBolck>>>(pb->grad_x,pc->grad_x,pa->grad_x,(pa->row)*(pa->column));  
    }
}


void MY_GRAPH_NET :: dividing_point(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE **pb, MY_MATRIX_DEVICE **pc)
{
    *pb = new MY_MATRIX_DEVICE;
    *pc = new MY_MATRIX_DEVICE;
   
    deque_matrix.AddLast(*pb);
    deque_matrix.AddLast(*pc);
    
    (*pb)->row = pa->row;
    (*pb)->column = pa->column;
    (*pc)->row = pa->row;
    (*pc)->column = pa->column;

    deque_operate.AddLast(pa,*pb,*pc,&gate_dividing_point);
}



void gate_stack(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,(pa->row)*(pa->column));  
        blocksPerGride = (pb->row*pb->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>((pc->x)+((pa->row)*(pa->column)),pb->x,(pb->row)*(pb->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,(pa->row)*(pa->column));  
        blocksPerGride = (pb->row*pb->column + threadsPerBolck -1)/threadsPerBolck;
        transfer<<<blocksPerGride, threadsPerBolck>>>(pb->grad_x,(pc->grad_x)+((pa->row)*(pa->column)),
                (pb->row)*(pb->column));  


    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: stack(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb)
{
    MY_FUNC_ERROR(pa->row == pb->row);

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column + pb->column;

    deque_operate.AddLast(pa,pb,pc,&gate_stack);
    return pc;

}


void gate_merge(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        merge<<<blocksPerGride, threadsPerBolck>>>(pc->x,pc->row,
                pa->x,pa->row,0,(pa->row)*(pa->column));  
  
        blocksPerGride = (pb->row*pb->column + threadsPerBolck -1)/threadsPerBolck;
        merge<<<blocksPerGride, threadsPerBolck>>>(pc->x,pc->row,
                pb->x,pb->row, pa->row,(pb->row)*(pb->column));  
    }
    else if(stat == BACKWARD)
    {
      printf("sdfssdf\n");
       
      blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        inv_merge<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pa->row,
                pc->grad_x,pc->row,0,(pa->row)*(pa->column));  
        
        
        blocksPerGride = (pb->row*pb->column + threadsPerBolck -1)/threadsPerBolck;
        inv_merge<<<blocksPerGride, threadsPerBolck>>>(pb->grad_x,pb->row,
                pc->grad_x,pc->row,pa->row, (pb->row)*(pb->column));  


    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: merge(MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb)
{
    MY_FUNC_ERROR(pa->column == pb->column);

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row + pb->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,pb,pc,&gate_merge);
    return pc;

}

void gate_rand_scale(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
     

    if((stat == FOREWARD) || (stat == TEST))
    {
   
        float from = pa->rate;
        float to = pc->rate;
        
        float temp = (rand()%(int)((to - from)*1000))/1000.0 + from;

        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        multi_scala<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,temp,(pa->row)*(pa->column)); 

        pa->rate = temp;
    }
    else if(stat == BACKWARD)
    {
        float temp = pa->rate;
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        multi_scala<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,temp,(pa->row)*(pa->column));  
   
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: rand_scale(MY_MATRIX_DEVICE *pa, float from, float to)
{

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);
    pa->rate = from;
    pc->rate = to;
    pc->row = pa->row;
    pc->column = pa->column;

    deque_operate.AddLast(pa,NULL,pc,&gate_rand_scale);
    return pc;


}

void gate_scale(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    float scale = pa->rate;

    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        multi_scala<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,scale,(pa->row)*(pa->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        multi_scala<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,scale,(pa->row)*(pa->column));  
   
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: scale(MY_MATRIX_DEVICE *pa, float fa)
{

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;
    
    pa->rate = fa;
    deque_operate.AddLast(pa,NULL,pc,&gate_scale);
    return pc;


}


void gate_uniform_noise(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    if((stat == FOREWARD) || (stat == TEST))
    {
        CURAND_CALL(curandGenerateUniform(((MY_GRAPH_NET*)pthis)->rand_gen,pc->x,(pc->row)*(pc->column)));
    }
}


MY_MATRIX_DEVICE* MY_GRAPH_NET :: uniform_noise(int row, int column)
{

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = row;
    pc->column = column;

    deque_operate.AddLast(NULL,NULL,pc,&gate_uniform_noise);
    return pc;


}

void gate_white_noise(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 

    if((stat == FOREWARD) || (stat == TEST))
    {
        CURAND_CALL(curandGenerateUniform(((MY_GRAPH_NET*)pthis)->rand_gen,pc->x,(pc->row)*(pc->column)));
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        dropout_table<<<blocksPerGride, threadsPerBolck>>>(pc->x,pc->rate,(pc->row)*(pc->column));  
 

    }
}



MY_MATRIX_DEVICE* MY_GRAPH_NET :: white_noise(int row, int column,float rate)
{
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = row;
    pc->column = column;
    pc->rate = rate;
    deque_operate.AddLast(NULL,NULL,pc,&gate_white_noise);
    return pc;


}


void gate_min(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    if((stat == FOREWARD) || (stat == TEST))
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        min<<<blocksPerGride, threadsPerBolck>>>(pc->x,pa->x,pc->rate,(pa->row)*(pa->column));  
    }
    else if(stat == BACKWARD)
    {
        blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
        min_inv<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pa->x,pc->rate,(pa->row)*(pa->column));  
      
        blocksPerGride = (pc->row*pc->column + threadsPerBolck -1)/threadsPerBolck;
        scalar_multi<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,pa->grad_x,(pc->row)*(pc->column));  
    }
}




MY_MATRIX_DEVICE* MY_GRAPH_NET :: min(MY_MATRIX_DEVICE *pa, float max_value)
{
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;
    pc->rate = max_value;
    deque_operate.AddLast(pa,NULL,pc,&gate_min);
    return pc;

}

float MY_GRAPH_NET :: sum_absolute(MY_MATRIX_DEVICE *pa)
{
    float result;

    CUBLAS_CALL(cublasSasum(handle,(pa->row)*(pa->column),pa->x,1,&result));
    return result;
}


float MY_GRAPH_NET :: average_absolute(MY_MATRIX_DEVICE *pa)
{
    float result;

    CUBLAS_CALL(cublasSasum(handle,(pa->row)*(pa->column),pa->x,1,&result));
    return result/((pa->row)*(pa->column));
}

float MY_GRAPH_NET :: accuracy(MY_MATRIX_DEVICE *y, MY_MATRIX_DEVICE *t)
{
    MY_FUNC_ERROR((y->row == t->row)&&(y->column == t->column));


    
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    float result;
    
    blocksPerGride = (y->column + threadsPerBolck -1)/threadsPerBolck;
    accuracy_table<<<blocksPerGride, threadsPerBolck>>>(y->x,t->x,d_temp.x,y->row, y->column);  
    
    CUBLAS_CALL(cublasSasum(handle,y->column,d_temp.x,1,&result));
    
    return result/y->column;
}


//-----------------------------

MY_MOMENTUM_OPTIMIZER :: MY_MOMENTUM_OPTIMIZER()
{
    learning_rate = 0.1;
    momentum_rate = 0.0;

}
MY_MOMENTUM_OPTIMIZER :: ~MY_MOMENTUM_OPTIMIZER()
{

}
void MY_MOMENTUM_OPTIMIZER :: set_hyperpara(float l_rate, float m_rate)
{
    learning_rate = l_rate;
    momentum_rate = m_rate;
}



void gate_momentum_update(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    float l_rate = ((MY_MOMENTUM_OPTIMIZER*)pthis)->learning_rate;
    float m_rate = ((MY_MOMENTUM_OPTIMIZER*)pthis)->momentum_rate;



    blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
    momentum_vector<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,l_rate,m_rate,(pa->row)*(pa->column));  
    
    blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
    add<<<blocksPerGride, threadsPerBolck>>>(pa->x,pc->grad_x,pa->x,(pa->row)*(pa->column));  

}



void MY_MOMENTUM_OPTIMIZER :: set_para(MY_MATRIX_DEVICE *pa, ...)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
 
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = pa->row;
    pc->column = pa->column;
    CUDA_CALL(cudaMalloc(&(pc->grad_x),sizeof(float)*(pc->row)*(pc->column)));
    blocksPerGride = ((pc->row)*(pc->column) + threadsPerBolck -1)/threadsPerBolck; 
    make_zeros<<<blocksPerGride, threadsPerBolck>>>(pc->grad_x,(pc->row)*(pc->column));  


    deque_operate.AddLast(pa,NULL,pc,&gate_momentum_update);

    
    va_list ap;
    MY_MATRIX_DEVICE *arg;
    va_start(ap,pa);
    while(1){
        arg=va_arg(ap,MY_MATRIX_DEVICE*);
        if (arg == NULL) break;

        pc = new MY_MATRIX_DEVICE;
        deque_matrix.AddLast(pc);

        pc->row = arg->row;
        pc->column = arg->column;
        CUDA_CALL(cudaMalloc(&(pc->grad_x),sizeof(float)*(pc->row)*(pc->column)));
        blocksPerGride = ((pc->row)*(pc->column) + threadsPerBolck -1)/threadsPerBolck; 
        make_zeros<<<blocksPerGride, threadsPerBolck>>>(pc->grad_x,(pc->row)*(pc->column));  


        deque_operate.AddLast(arg,NULL,pc,&gate_momentum_update);


    }
    va_end(ap);

}
void MY_MOMENTUM_OPTIMIZER :: update()
{
    _node_graph_net *cur = deque_operate.head;
    do{
        cur->operate(this,cur->in1,cur->in2,cur->out,BACKWARD);
        cur = cur->next;
    }
    while(cur != NULL);


}

//----------------------------------------------

MY_ADAM_OPTIMIZER :: MY_ADAM_OPTIMIZER()
{
    learning_rate = 0.0001;
    beta1 = 0.9;
    beta2 = 0.999;
    beta1_t = beta1;
    beta2_t = beta2;


}
MY_ADAM_OPTIMIZER :: ~MY_ADAM_OPTIMIZER()
{

}
void MY_ADAM_OPTIMIZER :: set_hyperpara(float l_rate,float beta1_rate, float beta2_rate)
{
    learning_rate = l_rate;
    beta1 = beta1_rate;
    beta2 = beta2_rate;
    beta1_t = beta1;
    beta2_t = beta2;;


}

void gate_adam_update(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb,
        MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    float learning_rate = ((MY_ADAM_OPTIMIZER*)pthis)->learning_rate;
    float beta1 = ((MY_ADAM_OPTIMIZER*)pthis)->beta1;
    float beta2 = ((MY_ADAM_OPTIMIZER*)pthis)->beta2;
    float beta1_t = ((MY_ADAM_OPTIMIZER*)pthis)->beta1_t;
    float beta2_t = ((MY_ADAM_OPTIMIZER*)pthis)->beta2_t;

    blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
    adam_beta1<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pb->grad_x,beta1,(pa->row)*(pa->column));  
  
    blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
    adam_beta2<<<blocksPerGride, threadsPerBolck>>>(pa->grad_x,pc->grad_x,beta2,(pa->row)*(pa->column));  
    
    blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
    adam_sum<<<blocksPerGride, threadsPerBolck>>>(pb->grad_x,pc->grad_x,pa->x,learning_rate,
            beta1_t,beta2_t,(pa->row)*(pa->column));  

}



void MY_ADAM_OPTIMIZER :: set_para(MY_MATRIX_DEVICE *pa, ...)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    MY_MATRIX_DEVICE *pb = new MY_MATRIX_DEVICE;
    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pb);
    deque_matrix.AddLast(pc);

    pb->row = pa->row;
    pb->column = pa->column;
    CUDA_CALL(cudaMalloc(&(pb->grad_x),sizeof(float)*(pb->row)*(pb->column)));
    blocksPerGride = ((pb->row)*(pb->column) + threadsPerBolck -1)/threadsPerBolck; 
    make_zeros<<<blocksPerGride, threadsPerBolck>>>(pb->grad_x,(pb->row)*(pb->column));  

    pc->row = pa->row;
    pc->column = pa->column;
    CUDA_CALL(cudaMalloc(&(pc->grad_x),sizeof(float)*(pc->row)*(pc->column)));
    blocksPerGride = ((pc->row)*(pc->column) + threadsPerBolck -1)/threadsPerBolck; 
    make_zeros<<<blocksPerGride, threadsPerBolck>>>(pc->grad_x,(pc->row)*(pc->column));  


    deque_operate.AddLast(pa,pb,pc,&gate_adam_update);
    
    va_list ap;
    MY_MATRIX_DEVICE *arg;
    va_start(ap,pa);
    while(1){
        arg=va_arg(ap,MY_MATRIX_DEVICE*);
        if (arg == NULL) break;

        pb = new MY_MATRIX_DEVICE;
        pc = new MY_MATRIX_DEVICE;
        deque_matrix.AddLast(pb);
        deque_matrix.AddLast(pc);

        pb->row = arg->row;
        pb->column = arg->column;
        CUDA_CALL(cudaMalloc(&(pb->grad_x),sizeof(float)*(pb->row)*(pb->column)));
        blocksPerGride = ((pb->row)*(pb->column) + threadsPerBolck -1)/threadsPerBolck; 
        make_zeros<<<blocksPerGride, threadsPerBolck>>>(pb->grad_x,(pb->row)*(pb->column));  


        pc->row = arg->row;
        pc->column = arg->column;
        CUDA_CALL(cudaMalloc(&(pc->grad_x),sizeof(float)*(pc->row)*(pc->column)));
        blocksPerGride = ((pc->row)*(pc->column) + threadsPerBolck -1)/threadsPerBolck; 
        make_zeros<<<blocksPerGride, threadsPerBolck>>>(pc->grad_x,(pc->row)*(pc->column));  


        deque_operate.AddLast(arg,pb,pc,&gate_adam_update);

    }
    va_end(ap);
}

void MY_ADAM_OPTIMIZER :: update()
{
    _node_graph_net *cur = deque_operate.head;
    do{
        cur->operate(this,cur->in1,cur->in2,cur->out,BACKWARD);
        cur = cur->next;
    }
    while(cur != NULL);
    beta1_t = beta1_t * beta1;
    beta2_t = beta2_t * beta2;
}
//-------------------------------------------------

MY_REGULARIZATION :: MY_REGULARIZATION()
{
    L1_rate = 1e-8;
    L2_rate = 1e-8;
    max_rate = 2.0;


}
MY_REGULARIZATION :: ~MY_REGULARIZATION()
{

}

void gate_max_norm(void *pthis, MY_MATRIX_DEVICE *pa, MY_MATRIX_DEVICE *pb, MY_MATRIX_DEVICE *pc, GATE_STAT stat)
{
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    float rate = ((MY_REGULARIZATION*)pthis)->max_rate;
 
    blocksPerGride = (pa->row*pa->column + threadsPerBolck -1)/threadsPerBolck;
    max_norm<<<blocksPerGride, threadsPerBolck>>>(pa->x,rate,(pa->row)*(pa->column));  
}


void MY_REGULARIZATION :: set_para(REGULARIZATION_STAT stat, float rate,  ...)
{
    va_list ap;
    MY_MATRIX_DEVICE *arg;
    va_start(ap,rate);
    while(1){
        arg=va_arg(ap,MY_MATRIX_DEVICE*);
        if (arg == NULL) break;

        if(stat == MAX_NORM){
            max_rate = rate;
            deque_operate.AddLast(arg,NULL,NULL,&gate_max_norm);
        }
        else{
            printf("there are no L1,L2 norm");
            exit(1);
        } 
    }
    va_end(ap);

}
void MY_REGULARIZATION :: update()
{
    _node_graph_net *cur = deque_operate.head;
    do{
        cur->operate(this,cur->in1,cur->in2,cur->out,BACKWARD);
        cur = cur->next;
    }
    while(cur != NULL);

}

