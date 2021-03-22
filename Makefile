CFLAGS = -lm -lglut -lGL -lGLU
CC = g++

.PHONY: clean nbody test


clean:
	rm $(OBJECTS) a.out main.o

nbody: main.cpp
	$(CC) common.cpp render.cpp sequential-exhaustive.cpp main.cpp -o main.o $(CFLAGS)

run: nbody
	./main.o

