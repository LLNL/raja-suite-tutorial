=================================
Fractal Tutorial - HIP Execution
=================================

Before starting, be sure to study the CUDA/HIP implementations in the 02-CUDA and 
03-HIP directories before continuing. It is important to note:
 * Read-only, write-only, and read-write variables used in the main computation
 * The main data structure that holds the values of the fractal pixels
 * Any data dependencies, if any, throughout the computation of the pixels

A notable difference between launch and other RAJA API's is the support for run-time
backend selection. 

Look for the `TODO` comments in the source code. Here you will have to choose 
 RAJA host and device policies for the launch. 

A complete description of the different policies is available in the online RAJA
documentation:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#raja-loop-kernel-execution-policies

Once you are ready, uncomment the COMPILE define on on top of the file and do

```
$ make fractal-ex4-RAJA-HIP
$ ./bin/fractal-ex4-RAJA-HIP
```