
FLAGS =  -o

TARGET = main
OBJECTS = main.o my_mnist_class.o 





$(TARGET) : $(OBJECTS)
				$(CXX) $(FLAGS) $(TARGET) $(OBJECTS) -lpthread 


clean :
				rm *.o $(TARGET)

new :
				$(MAKE) clean
				$(MAKE)
