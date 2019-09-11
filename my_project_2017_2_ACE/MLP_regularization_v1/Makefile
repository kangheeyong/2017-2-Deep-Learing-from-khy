
NVCC = nvcc
FLAGS = -o


CUDA = MLP_basic.cu my_device_function.cu
CPP = my_mnist_class.cpp sub_fucntion.cpp
OBJECTS = MLP_basic.o my_mnist_class.o my_device_function.o sub_fucntion.o

LIB_CUDA = -lcublas -lcurand -Wno-deprecated-gpu-targets
LIB = -lpthread


test : main_test.cpp $(CPP) $(CUDA) 
				$(NVCC) -c main_test.cpp $(CPP) $(CUDA) $(LIB_CUDA) $(LIB)
				$(NVCC) $(FLAGS) test_experiment main_test.o $(OBJECTS) $(LIB_CUDA) $(LIB)


train : main_train.cpp $(CPP) $(CUDA) 
				$(NVCC) -c main_train.cpp $(CPP) $(CUDA) $(LIB_CUDA) $(LIB)
				$(NVCC) $(FLAGS) train_experiment main_train.o $(OBJECTS) $(LIB_CUDA) $(LIB)




validation : main_validaion.cpp $(CPP) $(CUDA) 
				$(NVCC) -c main_validaion.cpp $(CPP) $(CUDA) $(LIB_CUDA) $(LIB)
				$(NVCC) $(FLAGS) validation_experiment main_validaion.o $(OBJECTS) $(LIB_CUDA) $(LIB)








clean :
				rm *.o 

new :
				$(MAKE) clean
				$(MAKE)
