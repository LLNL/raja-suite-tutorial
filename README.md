# RAJA Portability Suite Tutorial Series

Welcome to the RAJA Portability Suite tutorial series. In this repo, you
will find a set of tutorials. This set includes an introductory tutorial
which will provide guided set of lessons to follow to learn how to get
started with the RAJA Portability Suite. The set also includes a more
advanced (i.e. Intermediate) tutorial meant for those who would like more 
hands-on instruction for RAJA. We would suggest starting with the Intro 
tutorial and then moving on to the Intermediate tutorial.

If you are running on an LC machine and would like to build locally, 
be sure to do the following to build and run the tutorials:
```
module load cmake/3.20.2
module load gcc/8.3.1
module load cuda/11.2.0
cmake -DENABLE_CUDA=On -DENABLE_OPENMP=Off -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-11.2.0/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-11.2.0 -DBLT_CXX_STD=c++14 -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_EXERCISES=On -DRAJA_ENABLE_OPENMP=Off -DCMAKE_CUDA_FLAGS=--extended-lambda -DCUDA_ARCH=sm_70 ../
```

(Note: you need a cmake version greater than 3.19 and you need a more
recent gcc version which can handle c++14.)

# License

This tutorial is licensed under the BSD 3-Clause license.

Copyrights and patents in the RAJA project are retained by contributors. No
copyright assignment is required to contribute to RAJA.

Unlimited Open Source - BSD 3-clause Distribution 

LLNL-CODE-689114 
OCEC-16-063

For release details and restrictions, please see the information in the following:

LICENSE
NOTICE
