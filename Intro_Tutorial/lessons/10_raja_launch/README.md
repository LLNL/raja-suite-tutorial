# Lesson ten

In this lesson we begin exploring the RAJA abstraction for exposing parallism
within nested loops and the GPU thread/block programming model.

Prior to this lesson we have only explored the `RAJA::forall` method.
Although easy to use, the ``RAJA::forall`` method is limited creating
a single parallel region and distributing the iterations within a for
loop to threads via OpenMP or a GPU execution policy.

The ``RAJA::launch`` API as demonstrated below is designed to create a kernel execution
space in which developers may express their algorithms in terms of nested ``RAJA::loop``
methods. Similar to GPU programming models a compute grid can be configured using the
``RAJA::Teams`` and ``RAJA::Threads`` constructs in the ``RAJA::LaunchParams`` container.

``RAJA::loop`` methods can be excuted sequentially using ``RAJA::seq_exec``.
Dispatching loop iterations from CUDA threads and blocks can be done using the following
policies shown here for the x-dimension (similar policies exist for the y,z-dimensions).

| RAJA                                        |       CUDA                              |
|---------------------------------------------|---------------------------------------- |
| RAJA::cuda_thread_direct_x                  | threadIdx.x                             |
| RAJA::cuda_block_direct_x                   | blockIdx.x                              |
| RAJA::cuda_global_size_x_direct<block_size> | blockIdx.x * blockDim.x  + threadIdx.x  |


```
  RAJA::launch<EXEC_POL>(
    RAJA::LaunchParams(RAJA::Teams(teams), RAJA::Threads(team_size)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        // ``RAJA::Loops`` may be nested within an kernel execution space
        RAJA::loop<LOOP_POL>(ctx, RAJA::TypedRangeSegment<int>(0,M), [&] (int row) {
          RAJA::loop<LOOP_POL>(ctx, RAJA::TypedRangeSegment<int>(0,M), [&] (int row) {
              // Computation
          });
        });

    });
```

In this final lesson we invite the particpants to complete the policy selection for a matrix-transpose example.
