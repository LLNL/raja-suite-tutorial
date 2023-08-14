===============================================
Fractal Tutorial - Base (Sequential) Execution
===============================================

This file has no exercises but it
is used as a base reference implementation.

Use the code to compare implementations
between the RAJA kernel and launch abstractions.

Compile and run the code:

```
$ make fractal-ex0-c-loop
$ ./bin/fractal-ex0-c-loop
```

Before starting, be sure to study the seq-exec implementation of the fractal. 
It is important to note:
 * Read-only, write-only, and read-write variables used in the main computation
 * The main data structure that holds the values of the fractal pixels
 * Any data dependencies, if any, throughout the computation of the pixels

Also note that this is a sequential implementation. Timing information will be output to the screen. As we add RAJA and Umpire, it will be interesting to see how performance improves.