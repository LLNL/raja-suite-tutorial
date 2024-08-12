# Lesson 10

In this lesson, you will learn how to use `RAJA::kernel` to write nested loops.

The previous lesson used multiple `RAJA::forall` calls, nested inside each
other, to implement a matrix multiplication. This pattern will work when
executing on the CPU, but not on a GPU. It is also less efficient. 

That's why RAJA provides the `RAJA::kernel` functionality. We can create a
`RAJA::KernelPolicy` to describe the layout of our nested loops. For example,
this triple nest loop on the CPU:

```
  for (int k = kmin; k < kmax; ++k) {
    for (int j = jmin; j < jmax; ++j) {
      for (int i = imin; i < imax; ++i) {
        printf( " (%d, %d, %d) \n", i, j, k);
      }
    }
  }
```

will require a kernel policy and kernel like this:

```
using KJI_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<2, RAJA::seq_exec,    // k
                          RAJA::statement::For<1, RAJA::seq_exec,  // j
                            RAJA::statement::For<0, RAJA::seq_exec,// i 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;

  RAJA::kernel<KJI_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (int i, int j, int k) { 
     printf( " (%d, %d, %d) \n", i, j, k);
  });
```

Where the IRange, JRange, and KRange are simply defined like:

```
RAJA::TypedRangeSegment<int> KRange(0, kmax);
RAJA::TypedRangeSegment<int> JRange(0, jmax);
RAJA::TypedRangeSegment<int> IRange(0, imax);
```

Take a look at the RAJA documentation for a detailed explanation of the
`RAJA::kernel` method:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/kernel_nested_loop_reorder.html

We have created the two ranges you need, and the body of the loops can be the
same as in the previous lesson. You will need to add the `RAJA::kernel` method
call, and define an execution policy `EXEC_POL` that will correctly execute the
kernel. 

If you are stuck, you can reference the matrix-multiply example in the RAJA
repository:
https://github.com/LLNL/RAJA/blob/develop/examples/tut_matrix-multiply.cpp

Keep in mind that this matrix multiplication lesson is built upon the previous
dot product lessons since a matrix multiplication is just a matrix version of
the dot product. The `RAJA::View` help us see this connection better.

When you have finished making your changes, uncomment the COMPILE define on line 7;
then compile and run the code:

```
$ make raja_kernel
$ ./bin/raja_kernel
```
