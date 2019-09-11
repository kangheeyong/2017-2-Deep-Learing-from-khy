#include "sub_fucntion.h"







int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch(devProp.major)
    {
        case 2 ://Fermi
            if (devProp.major == 1) cores = mp*48;
            else cores = mp*32;
            break;
        case 3 : //Kepler
            cores = mp*192;
            break;
        case 5 : //Maxwell
            cores = mp*128;
            break;
        case 6 : //Pascal
            if(devProp.minor == 1) cores = mp *128;
            else if(devProp.minor == 0) cores = mp*64;
            else printf("Unknown device type\n");
            break;
        defualt :
            printf("Unknown device type\n");
            break;

    }
    return cores;
}
bool ChoseGpuAvailable(int n)
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);

    cout<<"devicesCount : "<<devicesCount<<endl;
    
    for(int i = 0 ; i < devicesCount ; i++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties,i);
        cout<<"----- device "<<i<<" -----"<<endl;
        cout<<"device name : "<<deviceProperties.name<<endl;
        cout<<"clock rate : "<<deviceProperties.clockRate/1048576.0<<" GHz"<<endl;
        cout<<"cores : "<<getSPcores(deviceProperties)<<endl;
        cout<<"totalGlobalMem : "<<deviceProperties.totalGlobalMem/1073741824.0<<" GByte"<<endl;

    }
    cout<<endl;
    if(n > devicesCount && n < 0) return false;
    else
    {
        cudaSetDevice(n);

        return true;
    }
}


static int *my_index;
static int max_my_index;
static int cur_point;

void make_index(int max_size)
{
    if(my_index != NULL) free(my_index);

    max_my_index = max_size;
    my_index = (int*)calloc(max_my_index,sizeof(int));
    for(int i = 0 ; i < max_my_index ; i++) my_index[i] = i;
}

void  shuffle_index()
{
    int temp;
    int cur;
    for(int i = 0 ; i < max_my_index ; i++)
    {
        cur = rand()%max_my_index;
        temp = my_index[i];
        
        my_index[i] = my_index[cur];
        my_index[cur] = temp;
    }
}

int get_index(int n)
{
    return my_index[n];
}

int get_next()
{
    cur_point = (cur_point+1) % max_my_index;
    return my_index[cur_point];
}







