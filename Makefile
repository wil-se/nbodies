CFLAGS = -lm -lglut -lGL -lGLU
CC = g++

.PHONY: clean nbody test


clean:
	rm $(OBJECTS) a.out main.o

nbody: main.cpp
	$(CC) common.cpp sequential-exhaustive.cpp sequential-barneshut.cpp main.cpp render.cpp -o main.o $(CFLAGS)

run: nbody
	./main.o