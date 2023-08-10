# Lesson 11

In this lesson, you will learn to convert the matrix multiplication
`RAJA::kernel` example to use CUDA.

This lesson requires some background understanding of how you can map work to
the CUDA programming model concepts of threads and blocks. This is currently
outside the scope of this tutorial.

A complete description of the different policies is available in the online RAJA
documentation:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/kernel_exec_pols.html#

Note that `RAJA_DEVICE` in the `RAJA::kernel` needs to be uncommented
before this will work!

Once you are ready, uncomment the COMPILE define on line 7; then you can build and run the example:

```
$ make eleven
$ ./bin/eleven
```

For reference, lesson 12 contains the solution, so don't worry if you get stuck!
