#!/usr/bin/zsh

FILES="main.cpp"

while [[ 1 ]]; do inotifywait -qe modify $FILES; eval "make run < input" ; done

