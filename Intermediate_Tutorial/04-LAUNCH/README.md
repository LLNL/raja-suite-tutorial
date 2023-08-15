=================================
Fractal Tutorial - LAUNCH Execution
=================================

The RAJA launch API introduces the concept of an execution space enabling
developers to express algorithms in terms of nested RAJA::loops. As the kernel
execution space is exposed to developers, static shared memory is avaible when
using the CUDA/HIP backends. The launch abstraction also takes a more explicit
approach in configuring device compute grid parameters. Finally, RAJA launch
can take both a host and device execution policy enabling run-time dispatch selection.

Look for the `TODO` comments in the source code. The main task here is to select
a host and device policy for the launch configuration and loop function.

A complete description of the different policies is available in the online RAJA
documentation:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#raja-loop-kernel-execution-policies

Once you are ready, uncomment the COMPILE define on on top of the file and do

```
$ make fractal-ex4-RAJA-HIP
$ ./bin/fractal-ex4-RAJA-HIP
```