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
    if(argc != 2)
    {
        printf("USAGE : %s [number_of_item]\n",argv[0]);
        return 0;
    }

    FILE *fd;
    int magic_number;
    int number_of_images;
    int number_of_rows;
    int number_of_columns;
    unsigned char *buff;
    
    int item = atoi(argv[1]);


    fd = fopen("../MNIST_data/train-images.idx3-ubyte","r");
    if (fd == NULL)
    {
        printf("file read fail\n");
        return 0;
    }
    
    fread((unsigned char*)&magic_number,sizeof(int),1,fd);
    fread((unsigned char*)&number_of_images,sizeof(int),1,fd);
    fread((unsigned char*)&number_of_rows,sizeof(int),1,fd);
    fread((unsigned char*)&number_of_columns,sizeof(int),1,fd);
    
    magic_number = reverse_int(magic_number);
    number_of_images = reverse_int(number_of_images);
    number_of_rows = reverse_int(number_of_rows);
    number_of_columns = reverse_int(number_of_columns);


    printf("magic_number : %d, number_of_item : %d \n",magic_number,number_of_images);
    printf("rows : %d, columns : %d \n",number_of_rows,number_of_columns);
   
    int total_size = number_of_images * number_of_rows * number_of_columns;

    buff = (unsigned char*)malloc(sizeof(unsigned char)*total_size);

    fread(buff,sizeof(unsigned char),total_size,fd);
    

    int width = 28;
    int height = 28;
    int depth = 8;
    int channels = 1;

    IplImage *img = cvCreateImage(cvSize(width,height),depth,channels);
    
    memcpy(img->imageData,buff +item*784,784);    
    

    cvShowImage("sdfsdf",img);
    
    cvWaitKey(0);

    cvReleaseImage(&img);
 
    free(buff);
    fclose(fd);
    return 0;
} 
