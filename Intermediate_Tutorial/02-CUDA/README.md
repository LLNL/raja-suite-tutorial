=================================
Fractal Tutorial - CUDA Execution
=================================

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