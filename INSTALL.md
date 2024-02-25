
# Installation Guide of Differentiable Renderer and Custom Operators #


## Structure ## 

The code is structured in three main components:

1) The python-based DELIFFAS ([Projects](Projects))
2) The custom Tensorflow operators, which are implemented in C++/CUDA ([CustomTensorFlowCPPOperators](CustomTensorFlowCPPOperators))
3) The Cuda-based rasterizer ([CudaRenderer](CudaRenderer))

It is recommended to follow the installation description below *before* trying to run a project. Once this installation is done *all* projects should run without further installations required.

## Installation ## 

### Installation on the Linux servers: ###

**Anaconda / Python / Tensorflow Installation**
- Install the latest [Anaconda version](https://www.anaconda.com) 
- Create a conda environment: `conda create --name tf280 python=3.9`
- Activate your environment: `conda activate tf280`
- Install the cudatoolkit package: `conda install -c nvidia cudatoolkit=11.2.0`
- Install the cudatoolkit-dev package: `conda install -c conda-forge cudatoolkit-dev=11.2.2`
- Install tensorflow package: `pip install tensorflow==2.8.0`
- Tensorflow 2.8.0 does not natively finds the cuda package: To fix this do the following:
  - Open your bashrc_private: `vi ~/.bashrc_private`
  - Add the path to your cuda lib to the LD_LIBRARY_PATH variable. 
  - Example: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/HPS/RTMPC/work/Anaconda/envs/tf280/pkgs/cuda-toolkit/lib64`
  - Close the script and run `source ~/.bashrc_private`
- Now test the tensorflow:
  - Activate the new environment: `conda activate tf280`
  - Open python: `python`
  - Type: `import tensorflow as tf`
  - Type: `t=tf.constant(8)`
  - If no no-found errors are shown your conda/python/tensorflow is successfully installed
  
**Compiling the custom Tensorflow operators**
- Go to `CustomTensorFlowCPPOperators/cmakeTF2Linux`
- Add the OpenCV dirs for each partition, Tensorflow paths, and the nvcc path to your `.bashrc_private`. More precisely, define those variables: `TENSORFLOW_PATH`, `OPENCV_PATH_RECON`, `OPENCV_PATH_GPU20` `NVCC_PATH`
- Therefore, open this file with `vi ~/.bashrc_private`and add at the end of the file 
  - For example:
  - `export TENSORFLOW_PATH=/HPS/RTMPC/work/Programs/Anaconda3-4.1.1/envs/tf220/lib/python3.5/site-packages/tensorflow`
  - `export OPENCV_PATH_GPU20=/HPS/RTMPC/work/MultiViewSupervisionNRTracking/Code/CustomTensorFlowCPPOperators/thirdParty/linux/opencv-3.4.7/build2`
  - `export NVCC_PATH=/HPS/RTMPC/work/Anaconda/envs/tf280/pkgs/cuda-toolkit/bin/nvcc`
  - Or wherever the installations are in your case. Small note: the OpenCV links defined here should work for all users.
- Make sure to revise CUDA_ARCH in CMakeLists.txt to match your system architecture.
- Run `bash submitBuild.sh`. 
- Run `bash submitCompile.sh`. 
- If everything goes well you should find a file named `libCustomTensorFlowOperators.so` in `CustomTensorFlowCPPOperators/binaries/Linux/ReleaseGpu20`

**Compiling the CUDA-based rasterizer**
- Go to `CudaRenderer/cpp/cmakeTF2Linux`
- Make sure to revise CUDA_ARCH in CMakeLists.txt to match your system architecture.
- Run `bash submitBuild.sh`. 
- Run `bash submitCompile.sh`. 
- If everything goes well you should find a file named `libCudaRenderer.so` in `CudaRenderer/cpp/binaries/Linux/ReleaseGpu20` 

**Congratulations, you can now go to the `Projects` folder and run the python-based code.**

## Running a project ##

For running a project please enter the [Projects](Projects) directory and follow the instructions.
**Note:** Make sure that you activate your environment *before* trying to run one of the python scripts.
