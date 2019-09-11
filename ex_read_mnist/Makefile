
FLAGS = -o
TARGET = test
OBJECTS = read_train_images.o

$(TARGET) : $(OBJECTS)
				$(CXX) $(FLAGS) $(TARGET) $(OBJECTS) `pkg-config --libs opencv`

clean :
				rm *.o $(TARGET)

new :
				$(MAKE) clean
				$(MAKE)
