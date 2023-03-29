=================================
Fractal Tutorial - LOOP Execution
=================================

Before starting, be sure to study the loop-exec implementation of the fractal 
before continuing. It is important to note:
 * Read-only, write-only, and read-write variables used in the main computation
 * The main data structure that holds the values of the fractal pixels
 * Any data dependencies, if any, throughout the computation of the pixels

Look for the `TODO` comments in the source code. This is where you will need to fill in
what's needed. You will need to create an Umpire allocator for the fractal and
complete the appropriate `RAJA::Kernel` statement using the `RAJA::loop_exec` execution
policy.

The `loop_exec` policy is a good first step because it allows us to get a sense of the
performance using serial nested loops. From here, we have a good baseline to compare against
when transitioning to CUDA, HIP, etc. 

The `loop_exec` is better than just the `seq_exec` policy because it allows the compiler to 
generate any optimizations that its heuristics deem beneficial.
To learn more about the `loop_exec` RAJA execution policy, see `here <https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html?highlight=loop_exec#raja-loop-kernel-execution-policies>`_.
