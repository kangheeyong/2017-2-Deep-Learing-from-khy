.SUFFIXES : .cu




NVCC = nvcc

OBJETS =  my_graph_net.o my_graph_net_sub.o my_device_func.o sub_main.o my_mnist_class.o

LIB = -lcublas -lcurand -Wno-deprecated-gpu-targets


main_test : main_test.o $(OBJETS)
				$(NVCC) -o $@ $^ $(LIB)




clean :
				rm *.o 


.cu.o :
				$(NVCC) -c $< $(LIB)

.cpp.o : 
				$(NVCC) -c $< $(LIB)
