#include "my_mnist_class.h"



using namespace std;



int reverse_int(int i)
{
    unsigned char c1,c2,c3,c4;
    
    c1 = i&255;
    c2 = (i>>8)&255;
    c3 = (i>>16)&255;
    c4 = (i>>24) & 255;

    return ((int)c1<<24) + ((int)c2<<16) + ((int)c3<<8) + c4;
}



EXAMPLE_MNIST_CLASS :: EXAMPLE_MNIST_CLASS()
{
    fd_t10k_images = NULL;
    fd_t10k_labels = NULL;
    fd_train_images = NULL;
    fd_train_labels = NULL;
    number_of_batch = 1;
    cur_index = 0;
    cur_point = 0;

   
    buff = NULL;
    train_images = NULL;
    train_labels = NULL;
    validation_images = NULL;
    validation_labels = NULL;
    test_images = NULL;
    test_labels = NULL;
    
}


EXAMPLE_MNIST_CLASS :: ~EXAMPLE_MNIST_CLASS()
{
    if(fd_t10k_images != NULL) fclose(fd_t10k_images);
    if(fd_t10k_labels != NULL) fclose(fd_t10k_labels);
    if(fd_train_images != NULL) fclose(fd_train_images);
    if(fd_train_labels != NULL) fclose(fd_train_labels);

    if(buff != NULL) free(buff);
    if(train_images != NULL) free(train_images);
    if(train_labels != NULL) free(train_labels);
    if(validation_images != NULL) free(validation_images);
    if(validation_labels != NULL) free(validation_labels);
    if(test_images != NULL) free(test_images);
    if(test_labels != NULL) free(test_labels);

 
}

void EXAMPLE_MNIST_CLASS :: first_read(const char* str)
{    
    char path[1000];

    strcpy(path,str);
    strcat(path,"/t10k-images.idx3-ubyte");
    fd_t10k_images = fopen(path,"r");
    if (fd_t10k_images == NULL)
    {
        printf("file read fail\n");
        exit(0);
    }
    printf("read : %s\n",path);


    strcpy(path,str);
    strcat(path,"/t10k-labels.idx1-ubyte");
    fd_t10k_labels = fopen(path,"r");
    if (fd_t10k_labels == NULL)
    {
        printf("file read fail\n");
        exit(0);
    }
    printf("read : %s\n",path);


    strcpy(path,str);
    strcat(path,"/train-images.idx3-ubyte");
    fd_train_images = fopen(path,"r");
    if (fd_train_images == NULL)
    {
        printf("file read fail\n");
        exit(0);
    }
    printf("read : %s\n",path);

    strcpy(path,str);
    strcat(path,"/train-labels.idx1-ubyte");
    fd_train_labels = fopen(path,"r");
    if (fd_train_labels == NULL)
    {
        printf("file read fail\n");
        exit(0);
    }
    printf("read : %s\n",path);



}

void EXAMPLE_MNIST_CLASS :: second_init(int batch_size)
{

    number_of_batch = batch_size;

    buff = (unsigned char*)calloc(20000*MAX_IMAGE_SIZE,sizeof(unsigned char));
    
    test_labels = (float*)calloc(10000*MAX_LABEL_NUM,sizeof(float));
    test_images = (float*)calloc(10000*MAX_IMAGE_SIZE,sizeof(float));
    validation_labels = (float*)calloc(5000*MAX_LABEL_NUM,sizeof(float));
    validation_images = (float*)calloc(5000*MAX_IMAGE_SIZE,sizeof(float));

    fseek(fd_t10k_labels,8,SEEK_SET);
    fread(buff,sizeof(unsigned char),10000,fd_t10k_labels);
    for(int i = 0 ; i < 10000 ; i++)
    {
       for(int j = 0 ; j < MAX_LABEL_NUM ; j++)
      {
          if( j == (int)buff[i]) test_labels[IDX2C(j,i,MAX_LABEL_NUM)] = 1.0;
          else  test_labels[IDX2C(j,i,MAX_LABEL_NUM)] = 0.0;
 
      }
    }
 
    fseek(fd_t10k_images,16,SEEK_SET);
    fread(buff,sizeof(unsigned char),10000*MAX_IMAGE_SIZE,fd_t10k_images);
    for(int i = 0 ; i < 10000*MAX_IMAGE_SIZE ; i++)
    {
        test_images[i] = (float)buff[i]/255.0;
    }
    

    fseek(fd_train_labels,8+MAX_TRAIN_INX,SEEK_SET);
    fread(buff,sizeof(unsigned char),5000,fd_train_labels);
    for(int i = 0 ; i < 5000 ; i++)
    {
      for(int j = 0 ; j < MAX_LABEL_NUM ; j++)
      {
          if( j == (int)buff[i]) validation_labels[IDX2C(j,i,MAX_LABEL_NUM)] = 1.0;
          else  validation_labels[IDX2C(j,i,MAX_LABEL_NUM)] = 0.0;
      }
    }
    
    fseek(fd_train_images,16+MAX_IMAGE_SIZE*MAX_TRAIN_INX ,SEEK_SET);
    fread(buff,sizeof(unsigned char),5000*MAX_IMAGE_SIZE,fd_train_images);
    for(int i = 0 ; i < 5000*MAX_IMAGE_SIZE ; i++)
    {
        validation_images[i] = (float)buff[i]/255.0;
    }
    
    train_labels = (float*)calloc(2*number_of_batch*MAX_LABEL_NUM,sizeof(float));
    train_images = (float*)calloc(2*number_of_batch*MAX_IMAGE_SIZE,sizeof(float));
}

void EXAMPLE_MNIST_CLASS :: third_read_train(int index)
{
  cur_index = index;
  cur_input = &train_images[cur_point*number_of_batch*MAX_IMAGE_SIZE]; 
  cur_target = &train_labels[cur_point*number_of_batch*MAX_LABEL_NUM]; 
  cur_point = (cur_point + 1)%2;

  fseek(fd_train_labels,8+cur_index,SEEK_SET);
  fread(buff,sizeof(unsigned char),number_of_batch,fd_train_labels);
  for(int i = 0 ; i < number_of_batch ; i++)
  {
      for(int j = 0 ; j < MAX_LABEL_NUM ; j++)
      {
          if( j == (int)buff[i]) cur_target[IDX2C(j,i,MAX_LABEL_NUM)] = 1.0;
          else  cur_target[IDX2C(j,i,MAX_LABEL_NUM)] = 0.0;
 
      }
  }

  fseek(fd_train_images,16+MAX_IMAGE_SIZE*cur_index ,SEEK_SET);
  fread(buff,sizeof(unsigned char),number_of_batch*MAX_IMAGE_SIZE,fd_train_images);
  for(int i = 0 ; i < number_of_batch*MAX_IMAGE_SIZE ; i++)
  {
      cur_input[i] = (float)buff[i]/255.0;
  }

  
  
/*

  for(int y = 0 ; y < 28 ; y++)
  {
      for(int x = 0 ; x < 28 ; x++)
      {
          if(cur_input[IDX2C(x,y,28)] > 0.5) cout<<"x ";
          else cout<<"  ";
      }
      cout<<endl;
  }
  for(int x = 0 ; x < 10 ; x++)
  {
      printf("%1.1f ",cur_target[x]);
  }
  cout<<endl;

  printf("%d %d\n",cur_input,cur_target);

  for(int y = 0 ; y < 28 ; y++)
  {
      for(int x = 0 ; x < 28 ; x++)
      {
          if(cur_input[((number_of_batch-1)*MAX_IMAGE_SIZE) + IDX2C(x,y,28)] > 0.5) cout<<"x ";
          else cout<<"  ";
      }
      cout<<endl;
  }
  for(int x = 0 ; x < 10 ; x++)
  {
      printf("%1.1f ",cur_target[x]);
  }
  cout<<endl; 
*/

}

