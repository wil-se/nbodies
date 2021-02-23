#!/usr/bin/zsh

while [[ 1 ]]; do inotifywait -qe modify main.c; eval "clear && gcc -g -lm main.c lib-host.c -o main.o && gdb ./main.o -batch -ex run" ; done
