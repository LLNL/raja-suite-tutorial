=================================
Fractal Tutorial - CUDA Execution
=================================

Before starting, be sure to study the sequential implementation in the 01-SEQ directory 
before continuing. It is important to note:
 * Read-only, write-only, and read-write variables used in the main computation
 * The main data structure that holds the values of the fractal pixels
 * Any data dependencies, if any, throughout the computation of the pixels

Look for the `TODO` comments in the source code. Here you will have to choose 
two RAJA CUDA policies for the kernel API.

A complete description of the different policies is available in the online RAJA
documentation:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/kernel_exec_pols.html#

Once you are ready, uncomment the COMPILE define on on top of the file and do

```
$ make fractal-ex2-RAJA-CUDA
$ ./bin/fractal-ex2-RAJA-CUDA
```