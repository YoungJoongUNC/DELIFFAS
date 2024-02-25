#!/bin/bash

# call your program here
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
cd ../build/Linux
make -j 32