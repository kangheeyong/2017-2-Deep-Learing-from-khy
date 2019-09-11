#include <stdio.h>
#include <opencv2/opencv.hpp>

int main()
{

    int width = 28;
    int height = 28;
    int depth = 8;
    int channels = 1;

    IplImage *img = cvLoadImage("lena.png",0);
    IplImage *dst = cvCreateImage(cvSize(width,height),depth,channels);

    printf("w : %d, h : %d, d : %d, c : %d\n",img->width,img->height,img->depth,img->nChannels );

    cvShowImage("sdf",img);
    cvShowImage("sdfsdf",dst);
    
    cvWaitKey(0);

    cvReleaseImage(&img);
    cvReleaseImage(&dst);
    return 0;
}

