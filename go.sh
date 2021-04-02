#!/usr/bin/zsh

FILES="main.cpp"

# while [[ 1 ]]; do ./generator.py > input && make run < input; done
./generator.py > input
[[ -f input ]] && {
        make run < input
}
