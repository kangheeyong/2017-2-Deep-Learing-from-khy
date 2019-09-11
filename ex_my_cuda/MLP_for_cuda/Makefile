
NVCC = nvcc
FLAGS = -o

TARGET = main

CUDA = MLP_basic.cu
CPP = main.cpp
OBJECTS = main.o MLP_basic.o

LIB_CUDA = -lcublas -Wno-deprecated-gpu-targets



$(TARGET) : 
				$(NVCC) -c $(CPP) $(CUDA) $(LIB_CUDA)
				$(NVCC) $(FLAGS) $(TARGET) $(OBJECTS) $(LIB_CUDA)



clean :
				rm *.o $(TARGET)

new :
				$(MAKE) clean
				$(MAKE)
