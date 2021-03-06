#!/usr/bin/zsh

FILES="main.c"

while [[ 1 ]]; do inotifywait -qe modify $FILES; eval "clear && gcc -g -lm main.c lib-host-common.c lib-host-barnes.c lib-host-ex.c -o main.o && gdb ./main.o -batch -ex run < test_solar_system" ; done
