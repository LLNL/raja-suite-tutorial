# Lesson 5

In this lesson, you will learn how to use a `RAJA::ReduceSum` object to perform
a safe parallel reduction in a vector dot product.  Different programming models
support parallel reduction operations differently, so RAJA provides portable
reduction types that make it easy to perform reduction operations in kernels.
For example, we can create a sum reduction variable `dot` for a data type
`TYPE` with the initial reduction value of `0.0`: 

```
RAJA::ReduceSum<REDUCTION_POLICY, TYPE> dot(0.0);
```

Although in this lesson, we only use `RAJA::ReduceSum`, RAJA provides 
other reduction types, such as `RAJA::ReduceMin` and `RAJA::ReduceMax`.
More information on RAJA reductions can be found at:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/reduction.html#

A reduction object takes two template parameters. The `REDUCTION_POLICY` and a
`TYPE`, which was mentioned above. The `REDUCTION_POLICY` controls how the 
reduction is performed, and must be compatible with the execution policy of 
the kernel where the reduction is used.

For example, if my `RAJA::forall` has an execution policy of `RAJA::omp_parallel_for_exec`,
then the reduction policy must be `RAJA::omp_reduce`. More documentation on 
compatible reduction and execution policies can be found here:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#reducepolicy-label

The second parameter, the `TYPE` parameter, is just the data type of the 
variable, such as `int`.

In the file `five.cpp`, follow the instruction in the `TODO` comment. Once 
you have filled in the correct reduction statement, compile and run:

```
$ make five
$ ./bin/five
```
