#INC = -I/opt/cuda/include 
#SM_ARCH ?= compute_35
#SM_CODE ?= sm_35
#ARCH = -gencode arch=$(SM_ARCH),code=$(SM_CODE)
CFLAGS = -lm -lglut -lGL -lGLU
CC = gcc
OBJECTS = main.o

.PHONY: clean nbody test

%.o: %.cpp
	$(CC) $< -o $@ $(CFLAGS) 

clean:
	rm $(OBJECTS) a.out

nbody: $(OBJECTS)
	#$(CC) main.cpp $(OBJECTS) $(CFLAGS) $(INC) -o main.o
	$(CC) main.cpp $(OBJECTS) $(CFLAGS)

run: nbody
	./main.o

