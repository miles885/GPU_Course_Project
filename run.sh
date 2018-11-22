#!/bin/bash

argc=$#

if ! [ $argc -eq 1 ]; then
    echo "Usage:"
    echo "run.sh fileName"

    exit
fi

make clean
make
./project.exe $1
