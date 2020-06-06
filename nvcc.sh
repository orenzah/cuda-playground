#!/bin/sh
x="$1"
PWD=$(pwd)
file=${x%.*}
docker run -v $PWD:/compile -it  nvidia/cuda:10.1-devel-ubuntu18.04 nvcc compile/$file.cu -o compile/$file
