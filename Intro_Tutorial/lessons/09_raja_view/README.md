# Lesson 9

In this lesson, you will learn how to use `RAJA::View` to simplify
multidimensional indexing in a matrix-matrix multiplication kernel.

As is commonly done for efficiency in C and C++, we have allocated the data for
the matrices as one-dimensional arrays. Thus, we need to manually compute the
data pointer offsets for the row and column indices in the kernel.

The `View` has two template parameters `RAJA::View< TYPE, LAYOUT>`. The `TYPE`
is the data type of the underlying data, and `LAYOUT` is a `RAJA::Layout` type that describes how the data is arranged.

The `Layout` takes two template parameters: `RAJA::Layout<DIM, TYPE>`. Here,
`DIM`  is the dimension of the layout, and `TYPE` is the type used to index into
the underlying data. For example, a 2D layout could be described as:

```
RAJA::Layout<2, int>
```

The default data layout ordering in RAJA is row-major, which is the convention
for multi-dimensional array indexing in C and C++. This means that the rightmost
index will be stride-1, the index to the left of the rightmost index will have
stride equal to the extent of the rightmost dimension, and so on.

When constructing a `View`, the first argument is the data pointer, and the
remaining arguments are those required by the `Layout` constructor. For example:

```
RAJA::View<double, RAJA::Layout<2, int>> view(data, N, N);
```

where `data` is a `double*`, and `N` is the size of each dimension. The size of
`data` should be at least `N*N`.

In the file `09_raja_view.cpp`, there are two `TODO` comments where you should create two
views, A, and R. R will be created via a permuted view with the same right-oriented layout
as A. Knowledge of `RAJA::make_permuted_view` is not required to complete this task, but
more information can be found here:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/view.html#make-permuted-view.
There are two `TODO` comments where you should complete the loop bounds, and fill
in A and R with their respective index values.
When you are ready, uncomment the COMPILE define on line 8; then you can compile and run the code:

```
$ make 09_raja_view
$ ./bin/09_raja_view
```

For more information on Views and Layouts, see the RAJA
documentation: https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/view_layout.html



