# Lesson 10

In this lesson you will learn how to use `RAJA::kernel` to write nested loops.

The previous lesson used multiple `RAJA::forall` calls, nested inside each
other, to implement a matrix multiplication. This pattern will work when
executing on the CPU, but not on a GPU. It is also less efficient. 

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

When you have finished making your changes, compile and run the code:

```
$ make ten
$ ./bin/ten
```
