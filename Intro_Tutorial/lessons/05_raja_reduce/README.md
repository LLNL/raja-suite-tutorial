# Lesson 5

In lesson 4, we looked at a data parallel loop kernel. A kernel with a data dependence 
between iterates is **not data parallel**. A data dependence occurs when multiple threads
(or tasks) could write to the same memory location at the same time. This is often called
a **race condition**. A race condition can cause an algorithm to produce **non-deterministic**,
for example order-dependent, results. 

Consider an attempt to write an OpenMP parallel kernel to computed the sum of the elements in an array:

```
double sum = 0;

#pragma omp parallel for
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
```

This kernel has a race condition since multiple loop iterates could write to the `sum`
variable at the same time. As a result, the computed `sum` value could be correct, but probably
not. OpenMP provides a reduction clause that allows a user to write a kernel to compute a
reduction without a race condition -- meaning only one thread writes to the reduction variable
at a time. For example:

```
double sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
```

This kernel will compute the sum in parallel correctly, although the results could be
non-deterministic due to order in which the values are accumulated in the sum.

It is important to note that not all parallel programming models provide a mechanism to write
a parallel reduction correctly. For example, CUDA and HIP do not. RAJA provides a reduction
construct allowing a user to write a parallel reduction in each of its supported programming
model back-ends. Although this lesson uses only `RAJA::ReduceSum`, RAJA provides other
reduction types, such as `RAJA::ReduceMin` and `RAJA::ReduceMax`. More information on
RAJA reductions can be found at:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/reduction.html#

In this lesson, you will learn how to use a `RAJA::ReduceSum` object to approximate
$\pi$, the ratio of the area in a circle over its diameter, using Riemann integrataion
and the formula
```math
\frac{ \pi }{4} = \tan^{-1}(1) = \int_0^1 \frac{1}{1+x^2}\,dx 
```
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
RAJA implementation of a kernel that approximates the value of $\pi$ using the 
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
