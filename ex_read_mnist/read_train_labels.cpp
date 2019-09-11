#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <stdlib.h>


int reverse_int(int i)
{
    unsigned char c1,c2,c3,c4;
    
    c1 = i&255;
    c2 = (i>>8)&255;
    c3 = (i>>16)&255;
    c4 = (i>>24) & 255;

    return ((int)c1<<24) + ((int)c2<<16) + ((int)c3<<8) + c4;
}



int main(int argc, char** argv)
{
    FILE *fd;
    int magic_number;
    int number_of_item;
    unsigned char *buff;

    fd = fopen("../MNIST_data/train-labels.idx1-ubyte","r");
    if (fd == NULL)
    {
        printf("file read fail\n");
        return 0;
    }
    
    fread((unsigned char*)&magic_number,sizeof(int),1,fd);
    fread((unsigned char*)&number_of_item,sizeof(int),1,fd);
    
    magic_number = reverse_int(magic_number);
    number_of_item = reverse_int(number_of_item);

    printf("magic_number : %d, number_of_item : %d \n",magic_number,number_of_item);
   

    buff = (unsigned char*)malloc(sizeof(unsigned char)*number_of_item);

    fread(buff,sizeof(unsigned char),number_of_item,fd);
    

    for(int i = 55000; i < 55020 ; i++)
    {
        printf("%d ",buff[i]);
    }
    printf("\n");
    
    free(buff);
    fclose(fd);
    return 0;
} 
