CFLAGS = -lm -lglut -lGL -lGLU -g -G
CC ?= nvcc

.PHONY: clean nbody test


clean:
	rm *.o

nbody: main.cu
	$(CC) -rdc=true -Xcompiler -fopenmp device-common.cu cuda-exhaustive.cu common.cu sequential-exhaustive.cu sequential-barneshut.cu main.cu render.cu -o main.o $(CFLAGS)

run: nbody
	./main.o 10
