# Lesson 5

In this lesson you will learn how to use a `RAJA::ReduceSum` to perform a safe
parallel reduction in a vector dot product.  Different programming models
support parallel reduction operations differently, so RAJA provides portable
reduction types that make it easy to perform reduction operations in kernels.

A reduction object takes two template parameters. The `REDUCTION_POLICY` and a
`TYPE`. The `REDUCTION_POLICY` controls how the reduction is performed, and must
be compatible with the execution policy of the loop where the reduction is used.
The `TYPE` parameter is just the data type of the variable.

```
RAJA::ReduceSum<REDUCTION_POLICY, TYPE> dot(0.0);
```