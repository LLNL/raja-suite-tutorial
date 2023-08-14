=================================
Fractal Tutorial - Base Execution
=================================

This file has no exercises but it
is used as a base reference implementation.

Use the code to compare implementations
between the RAJA kernel and launch abstractions.

To run the code compile and run via:

```
$ make fractal-ex0-c-loop
$ ./bin/fractal-ex0-c-loop
```

Before starting, be sure to study the seq-exec implementation of the fractal
before continuing. It is important to note:
 * Read-only, write-only, and read-write variables used in the main computation
 * The main data structure that holds the values of the fractal pixels
 * Any data dependencies, if any, throughout the computation of the pixels

Look for the `TODO` comments in the source code. This is where you will need to fill in
what's needed. You will need to create an Umpire pooled allocator (just like you did for
lesson 12 from the Introduction Tutorial) for the fractal and
complete the appropriate `RAJA::Kernel` statement using the `RAJA::seq_exec` execution
policy.

The `seq_exec` policy is a good first step because it allows us to get a sense of the
performance using serial nested seqs. From here, we have a good baseline to compare against
when transitioning to CUDA, HIP, etc.