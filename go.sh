#!/usr/bin/zsh

FILES="main.cpp"

while [[ 1 ]]; do ./generator.py > input && make run < input; done

