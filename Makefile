CFLAGS = -lm -lglut -lGL -lGLU
CC = nvcc
RCDIO = -gencode arch=compute_35,code=sm_35

.PHONY: clean nbody test


clean:
	rm $(OBJECTS) a.out main.o

nbody:
	$(CC) $(RCDIO) common.cpp sequential-exhaustive.cpp sequential-barneshut.cpp main.cpp render.cpp cuda-barneshut.cu -o main.o $(CFLAGS)

run: nbody
	./main.o
