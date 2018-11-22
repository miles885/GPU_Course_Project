#!/bin/bash

argc=$#

if ! [ $argc -eq 2 ]; then
    echo "Usage:"
    echo "run.sh numThreads blockSize"

    exit
fi

make clean
make
./assignment.exe $1 $2
