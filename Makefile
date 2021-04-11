CFLAGS = -lm -lglut -lGL -lGLU
CC = nvcc

.PHONY: clean nbody test


clean:
	rm *.o

nbody: main.cu
	$(CC) -rdc=true device-common.cu cuda-exhaustive.cu common.cu sequential-exhaustive.cu sequential-barneshut.cu main.cu render.cu -o main.o $(CFLAGS)

run: nbody
	./main.o