=================================
Fractal Tutorial - HIP Execution
=================================

The main purpose of this lesson is to demonstrate the performance portability of RAJA and Umpire. Up until now, we have been using the cuda_exec policy which is specific to NVIDIA GPUs and the CUDA API. Now, we will have to prepare our program for use on AMD GPUs with the HIP API. 

Note: Running this part of the code can be tricky because we will now have to run this on a different machine that is equipped with AMD GPUs. At LLNL, this will mean just ssh'ing into a different machine with the right backend hardware. However, if you don't have access to these types of machines, you can try porting this lesson to the openmp offload or SYCL execution policies (but that is beyond the scope of this tutorial).

Look for the `TODO` comments in the source code. Here you will have to choose 
two RAJA HIP policies for the kernel API.

A complete description of the different policies is available in the online RAJA
documentation:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/kernel_exec_pols.html#

Once you are ready, uncomment the COMPILE define on on top of the file and do

```
$ make fractal-ex3-RAJA-HIP
$ ./bin/fractal-ex3-RAJA-HIP
```