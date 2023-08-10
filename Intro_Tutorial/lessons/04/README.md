# Lesson Four

In this lesson, you will learn to write a loop using the `RAJA::forall` statement.

The `RAJA::forall` loop execution method is a template that takes an execution
policy type template parameter. A `RAJA::forall` method takes two arguments: an
iteration space object, such as a contiguous range of loop indices as shown
here, and a single lambda expression representing the loop kernel body:

```
RAJA::forall<EXEC_POLICY>( ITERATION SPACE, LAMBDA);
```

We can create a `RAJA::TypedRangeSegment` to describe an iteration space
that is a contiguous sequence of integers `[0, N)`.

```
RAJA::TypedRangeSegment<int>(0, N)
```

The lambda expression needs to take one argument, the loop index:

```
[=](int i) { // loop body }
```

the `[=]` syntax tells the lambda to capture arguments by value (e.g. create a
copy, rather than a reference).

The `EXEC_POLICY` template argument controls how the loop will be executed. In
this example, we will use the `RAJA::seq_exec` policy to execute this loop on
the CPU. In later lessons, we will learn about other policies that allow us to
run code on a GPU.

In the file four.cpp, you will see a `TODO` comment where you can add a 
`RAJA::forall` loop to initialize the array you allocated in the previous 
lesson.

When you have made your changes, compile and run the code in the same way as the
other lessons:

```
$ make four
$ ./bin/four
Address of data: 
data[50] = 50
```




