.SUFFIXES : .cu




NVCC = nvcc

OBJETS =  my_graph_net.o my_graph_net_sub.o my_device_func.o sub_main.o my_mnist_class.o

LIB = -lcublas -lcurand -Wno-deprecated-gpu-targets  `pkg-config --libs opencv`



main_test : main_test.o $(OBJETS)
				$(NVCC) -o $@ $^ $(LIB)


resNet_v1_test : resNet_v1.o $(OBJETS)
				$(NVCC) -o $@ $^ $(LIB)


resNet_v2_test : resNet_v2.o $(OBJETS)
				$(NVCC) -o $@ $^ $(LIB)

resNet_v3_test : resNet_v3.o $(OBJETS)
				$(NVCC) -o $@ $^ $(LIB)

resNet_v4_test : resNet_v4.o $(OBJETS)
				$(NVCC) -o $@ $^ $(LIB)

resNet_v5_test : resNet_v5.o $(OBJETS)
				$(NVCC) -o $@ $^ $(LIB)






clean :
				rm *.o 


.cu.o :
				$(NVCC) -c $< $(LIB)

.cpp.o : 
				$(NVCC) -c $< $(LIB)
