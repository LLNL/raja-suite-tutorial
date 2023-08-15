=================================
Fractal Tutorial - SEQ Execution
=================================

Before starting, be sure to study the base implementation in the 00-BASE directory.
Remember, it is important to note:
 * Read-only, write-only, and read-write variables used in the main computation
 * The main data structure that holds the values of the fractal pixels
 * Any data dependencies, if any, throughout the computation of the pixels

Look for the `TODO` comments in the source code. This is where you will need to fill in
what's needed. You will need to create an Umpire pooled allocator (just like you did for
lesson 12 from the Introduction Tutorial) for the fractal and
complete the appropriate `RAJA::Kernel` statement using the `RAJA::seq_exec` execution
policy.

A complete description of the different policies is available in the online RAJA
documentation:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#raja-loop-kernel-execution-policies

The `seq_exec` policy is a good first step because it allows us to get a sense of the
performance using sequential, nested loops with RAJA. 
From here, we have a good baseline to compare against when transitioning to 
CUDA, HIP, etc. 

To run the code compile and run via:

```
$ make fractal-ex1-RAJA-seq
$ ./bin/fractal-ex1-RAJA-seq
```
