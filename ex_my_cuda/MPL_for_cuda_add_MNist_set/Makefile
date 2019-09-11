
NVCC = nvcc
FLAGS = -o

TARGET = main

CUDA = MLP_basic.cu
CPP = main.cpp my_mnist_class.cpp
OBJECTS = main.o MLP_basic.o my_mnist_class.o

LIB_CUDA = -lcublas -lcurand -Wno-deprecated-gpu-targets
LIB = -lpthread


$(TARGET) : $(CPP) $(CUDA) 
				$(NVCC) -c $(CPP) $(CUDA) $(LIB_CUDA) $(LIB)
				$(NVCC) $(FLAGS) $(TARGET) $(OBJECTS) $(LIB_CUDA) $(LIB)



clean :
				rm *.o $(TARGET)

new :
				$(MAKE) clean
				$(MAKE)
