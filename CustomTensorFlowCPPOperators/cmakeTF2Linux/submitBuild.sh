#!/bin/bash

# call your program here
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
./createBuildLinux.sh --use-gpu ${CUDA_VISIBLE_DEVICES}