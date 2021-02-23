#!/usr/bin/zsh

while [[ 1 ]]; do inotifywait -qe modify main.c; eval "clear && nvcc main.c -rdc=true -o main.o && cuda-gdb ./main.o --batch --eval-command=run" ; done
