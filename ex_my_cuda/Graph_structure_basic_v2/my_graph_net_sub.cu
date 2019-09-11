#include "my_graph_net_sub.cuh"
#include "my_graph_net.cuh"
#include "my_device_func.cuh"
#include <string.h>
#include <stdarg.h>

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

MY_PARA_MANAGER :: MY_PARA_MANAGER()
{

}
MY_PARA_MANAGER :: ~MY_PARA_MANAGER()
{

}


MY_MATRIX_DEVICE* MY_PARA_MANAGER :: set(const char *name, int row, int column)
{

    MY_MATRIX_DEVICE *pc = new MY_MATRIX_DEVICE;
    deque_matrix.AddLast(pc);

    pc->row = row;
    pc->column = column;
    strcpy(pc->name,name);
    
    int const threadsPerBolck = 1024;
    int blocksPerGride = 0; 
    
    CUDA_CALL(cudaMalloc(&(pc->grad_x),sizeof(float)*(pc->row)*(pc->column)));
    blocksPerGride = ((pc->row)*(pc->column) + threadsPerBolck -1)/threadsPerBolck; 
    make_zeros<<<blocksPerGride, threadsPerBolck>>>(pc->grad_x,(pc->row)*(pc->column));  


    CUDA_CALL(cudaMalloc(&(pc->x),sizeof(float)*(pc->row)*(pc->column)));
    blocksPerGride = ((pc->row)*(pc->column) + threadsPerBolck -1)/threadsPerBolck; 
    make_zeros<<<blocksPerGride, threadsPerBolck>>>(pc->x,(pc->row)*(pc->column));  


    return pc;

}

void my_set_gaussian(float mean, float std, ...)
{
    va_list ap;
    MY_MATRIX_DEVICE *arg;
    curandGenerator_t rand_gen;
    CURAND_CALL(curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT));
 
       
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_gen,rand()));
    
    va_start(ap,std);
    while(1){
        arg=va_arg(ap,MY_MATRIX_DEVICE*);
        if (arg == NULL) break;
        CURAND_CALL(curandGenerateNormal(rand_gen,arg->x,(arg->row)*(arg->column),mean,std));
    
    }
    va_end(ap);
    
    CURAND_CALL(curandDestroyGenerator(rand_gen));
}

void my_para_write(const char *filename, ...)
{
    va_list ap;
    MY_MATRIX_DEVICE *arg;
    va_start(ap,filename);
    
    char filename_para[64];
    strcpy(filename_para,filename);
    *(strstr(filename_para,".txt")) = (char)NULL;


    FILE *fd_table,*fd_para;

    fd_table = fopen(filename,"w");
    fd_para = fopen(filename_para,"wb");

    MY_FUNC_ERROR(fd_table != NULL); 
    MY_FUNC_ERROR(fd_para != NULL);
    
    float *temp;
    while(1){
        arg=va_arg(ap,MY_MATRIX_DEVICE*);
        if (arg == NULL) break;
        fprintf(fd_table,"%s %d %d\n",arg->name,arg->row,arg->column);   
        temp = (float*)malloc(sizeof(float)*(arg->row)*(arg->column));
        CUDA_CALL(cudaMemcpy(temp,arg->x,sizeof(float)*(arg->row)*(arg->column),cudaMemcpyDeviceToHost));
        fwrite(temp,sizeof(float),(arg->row)*(arg->column),fd_para); 
        free(temp);
    }
    va_end(ap);
    
    fclose(fd_table);
    fclose(fd_para);
}
void my_para_read(const char *filename, ... )
{

    int cnt = 0;
    va_list ap;
    MY_MATRIX_DEVICE *arg,**arr_arg;
   
    va_start(ap,filename);
    while(1){
        arg=va_arg(ap,MY_MATRIX_DEVICE*);
        if (arg == NULL) break;
        cnt++;
    }
    va_end(ap);

   
    arr_arg = new MY_MATRIX_DEVICE* [cnt];
    int i = 0;
    va_start(ap,filename);
    while(1){
        arg=va_arg(ap,MY_MATRIX_DEVICE*);
        if (arg == NULL) break;
        arr_arg[i] = arg;
        i++;
    }
    va_end(ap);

   
    char filename_para[64];
    strcpy(filename_para,filename);
    *(strstr(filename_para,".txt")) = (char)NULL;

    FILE *fd_table,*fd_para;

    fd_table = fopen(filename,"r");
    fd_para = fopen(filename_para,"rb");

    MY_FUNC_ERROR(fd_table != NULL); 
    MY_FUNC_ERROR(fd_para != NULL);

    char para_name[64];
    int row,column; 
    float *temp;
    
    while( !feof(fd_table))            // 파일의 끝을 만난 때 까지 루프
    {
        arg = NULL;
        fscanf(fd_table,"%s %d %d\n",para_name,&row,&column);   

        for(i = 0 ; i < cnt ; i++)
        {
            if((strcmp(para_name,arr_arg[i]->name) == 0)&&(row == arr_arg[i]->row)&&(column == arr_arg[i]->column))
            {
                arg = arr_arg[i];
                break;
            }
        }

        MY_FUNC_ERROR(arg != NULL);
        temp = (float*)malloc(sizeof(float)*(arg->row)*(arg->column));
        fread(temp,sizeof(float),(arg->row)*(arg->column),fd_para); 
        CUDA_CALL(cudaMemcpy(arg->x,temp,sizeof(float)*(arg->row)*(arg->column),cudaMemcpyHostToDevice));
        free(temp);
    } 

    fclose(fd_table);
    fclose(fd_para);
    delete [] arr_arg;
}
void my_host2device(float *host, float* device, int n)
{
     CUDA_CALL(cudaMemcpy(device,host,sizeof(float)*n,cudaMemcpyHostToDevice));
       
}
void my_print(MY_MATRIX_DEVICE *pa)
{

    cout<<endl;
    float aaa[1000000];  
    int yy,xx;
    CUDA_CALL(cudaMemcpy(aaa,pa->x,sizeof(float)*(pa->row)*(pa->column),cudaMemcpyDeviceToHost));
    
    printf("%d %d\n",pa->row,pa->column);
    
    yy = pa->row;
    xx = pa->column;
   
    if(yy > 20)
    {
        yy = 20;
    }
    if(xx > 20)
    {
        xx = 20;
    }
    for(int y = 0 ; y < yy  ;y++){
        for(int x = 0 ; x < xx ;x++)
        {
            printf("%1.7f ",aaa[IDX2C(y,x,pa->row)]);
        }
        cout<<endl;
    }
    cout<<"---grad ---"<<endl;
    CUDA_CALL(cudaMemcpy(aaa,pa->grad_x,sizeof(float)*(pa->row)*(pa->column),cudaMemcpyDeviceToHost));
    
    for(int y = 0 ; y < yy ;y++){
        for(int x = 0 ; x < xx ;x++)
        {
            printf("%1.7f ",aaa[IDX2C(y,x,pa->row)]);
        }
        cout<<endl;
    }


}


void get_mnist_image(const char *name, MY_MATRIX_DEVICE *pa)
{

    int width = 28;
    int height = 28;
    int depth = 8;
    int channels = 1;
    char temp[100];

    float *float_data = new float[pa->row*pa->column];
    unsigned char *uch_data = new unsigned char[pa->row*pa->column];
    
    
    CUDA_CALL(cudaMemcpy(float_data,pa->x,sizeof(float)*(pa->row)*(pa->column),cudaMemcpyDeviceToHost));
    
    for(int i = 0 ; i < pa->row*pa->column ; i++)
    {
        uch_data[i] = (unsigned char)(float_data[i]*255.0);
    }

 
    IplImage *img = cvCreateImage(cvSize(width,height),depth,channels);
    
    for(int i = 0 ; i < pa->column ; i++)
    {    
        memcpy(img->imageData,uch_data+i*784,784); 
        sprintf(temp,"%s_%d.jpg",name,i);
        cvSaveImage(temp,img);
    }

    cvReleaseImage(&img);
 
}


