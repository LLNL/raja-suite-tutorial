# Lesson 5

In this lesson, you will learn how to use a `RAJA::ReduceSum` object to approximate
pi, the ratio of the area in a circle over its diameter. Some parallel programming models,
like OpenMP, provide a mechanism to perform a reduction in a kernel, while others like CUDA
do not. RAJA provides portable reduction types that make it easy to perform a reduction 
operation in a kernel that is syntactically similar for any programming model back-end that
RAJA supports.

Although this lesson uses only `RAJA::ReduceSum`, RAJA provides other reduction types,
such as `RAJA::ReduceMin` and `RAJA::ReduceMax`. More information on RAJA reductions can 
be found at:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/reduction.html#

We can create a sum reduction variable `pi` for a reduction policy and data type, and
initialize the reduction value to `0.0` as follows:

```
RAJA::ReduceSum<REDUCTION_POLICY, TYPE> pi(0.0);
```

The `REDUCTION_POLICY` type specializes the reduction object for use in a kernel with a
compatible execution policy. For example, if a `RAJA::forall` method uses an execution 
policy `RAJA::omp_parallel_for_exec`, then the reduction policy must be `RAJA::omp_reduce`.
Similarly, for CUDA, HIP, etc. More documentation on compatible reduction and execution
policies can be found at:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#reducepolicy-label

The second parameter, the `TYPE` parameter, is just the data type of the reduced quantity,
such as `int` or `double`.

The code for this lesson resides in the file `05_raja_reduce.cpp`. It provides a
RAJA implementation of a kernel that approximates the value of pi using the 
`RAJA::seq_exec` policy. With this policy, the loop will execute sequentially on a CPU.
The code will print out the result which should look familiar `3.145...`.
as expected. 

After that, you will find a `TODO` asking you to implement an OpenMP version of the kernel.
You will use `RAJA::omp_parallel_for_exec` for the execution policy to specialize the 
`RAJA::forall` template and `RAJA::omp_reduce` for the reduce policy to specialize the
`RAJA::ReduceSum` object.

Once you have filled in the correct reduction statement, compile and run:

```
$ make 05_raja_reduce
$ ./bin/05_raja_reduce
```

Are the results printed for the sequential and OpenMP kernels the same? Why or why not?

Try running the code multiple times by repeatedly executing:

```
$ ./bin/05_raja_reduce
```

What do you observe about the result printed for the sequential kernel for each run of
the code? What about the results printed for the OpenMP kernel for each run of the code?
Can you explain what is happening?
