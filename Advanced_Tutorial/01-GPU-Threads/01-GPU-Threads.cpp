#include "RAJA/RAJA.hpp"

  // GPU programming models such as CUDA and HIP peform computation
  // on a predefined grid composed of threads and blocks.
  // RAJA provides policies enabling users to utilize different
  // thread enumeration strategies. The two main strategies
  // are Block/Thread local enumeration and Global enumeration.
  //
  //
  //  Under a block and thread enumeration,
  //  threads have block local coordinate enumeration.
  //  While blocks are enumerated with respect
  //  to their location within the compute grid.
  //  The illustration below shows  2 x 2 compute grid
  //  wherein each block has 3 x 2 threads. Current
  //  programing models support up to three-dimensional
  //  block and thread configurations.
  //
  //
  //     Block (0,0)       Block (1,0)
  //   [0,0][0,1][0,2]  [0,0][0,1][0,2]
  //   [1,0][1,1][1,2]  [1,0][1,1][1,2]
  //
  //     Block (0,1)      Block (1,1)
  //   [0,0][0,1][0,2]  [0,0][0,1][0,2]
  //   [1,0][1,1][1,2]  [1,0][1,1][1,2]
  //
  //  Under the global enumeration each thread
  //  is a assigned a unique thread id based on
  //  on the dimension (2D illustrated here).
  //  The utility here comes when the iteration
  //  space is amendable to tiles in which blocks
  //  can be assigned to a tile and threads are
  //  assigned to work within a tile.
  //
  //   [0,0][0,1][0,2]  [0,3][0,4][0,5]
  //   [1,0][1,1][1,2]  [1,3][1,4][1,5]
  //
  //   [2,0][2,1][2,2]  [2,3][2,4][2,5]
  //   [3,0][3,1][3,2]  [3,3][3,4][3,5]
  //

  // Short note on RAJA nomenclature:
  // As RAJA serves as an abstraction layer
  // the RAJA::launch API uses the terms
  // teams and threads. Teams are analogous
  // to blocks in CUDA/HIP nomenclature
  // and workgroups in the SYCL programming model.
  // Threads are analogous to threads within CUDA/HIP
  // and work-items within the SYCL programming model.

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv))
{

#if defined(RAJA_ENABLE_CUDA)

  // The examples below showcase commonly used GPU policies.
  // For the HIP and SYCL programming models, we offer analogous policies.

  contexpr bool async = false; //asynchronous kernel execution

  using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async>>;

  // Example 1a. Block and thread direct polcies
  // Ideal for when iteration space can broken up into tiles
  // Teams can be assigned to a tile and threads can perform
  // computations within the tile

  // The example below employs the direct version for block
  // and thread policies, the underlying assumption is that
  // the loops are within the range of the grid and block sizes

  // In CUDA the direct loops are expressed as:
  //
  //   const int i = threadIdx.x;
  //   if(i < N) { //kernel }
  //
  {
    const int n_blocks = 50000;
    const int block_sz = 64;

    using outer_pol = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
    using inner_pol = RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;

    RAJA::launch<launch_policy>
      (RAJA::LaunchParams(RAJA::Teams(n_blocks), RAJA::Threads(block_sz)),
       [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

	RAJA::loop<outer_pol>(ctx, RAJA::RangeSegment(0, n_blocks), [&] (int bx) {


	    RAJA::loop<inner_pol>(ctx, RAJA::RangeSegment(0, block_sz), [&] (int tx) {

		//Do something

	      });
	  });


      });

  }

  // Example 1b. Block and thread loop polcies
  // Similar to the example above but using a thread loop
  // policy. The utility of the thread loop policy rises when
  // we consider multiple thread loops with varying iteration sizes.

  // If a RAJA loop iteration space is beyond the configured number
  // of threads in a team. The thread loop policies will perform a team
  // stride loop to span the whole range.

  // In CUDA the block stride loop is expressed as
  //
  //   for(int i=threadIdx.x; i<N; i+=blockDim.x)
  //
  {
    const int n_blocks = 50000;
    const int block_sz = 64;

    using outer_pol = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
    using inner_pol = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;

    RAJA::launch<launch_policy>
      (RAJA::LaunchParams(RAJA::Teams(n_blocks), RAJA::Threads(block_sz)),
       [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

	RAJA::loop<outer_pol>(ctx, RAJA::RangeSegment(0, n_blocks), [&] (int bx) {

	    //Iteration space is same as number of blocks per thread
	    //We could also use direct policy here
	    RAJA::loop<inner_pol>(ctx, RAJA::RangeSegment(0, block_sz), [&] (int tx) {
		//Do something here
	      }); //inner loop


	    //Iteration space is *more* than number of blocks per thread
	    RAJA::loop<inner_pol>(ctx, RAJA::RangeSegment(0, 2*block_sz), [&] (int tx) {
		//Do something here
	      }); //inner loop

	  }); //outer loop

      });

  }

  // Example 1c. Global Indexing
  // Main use case: Perfectly nested loops with large iteration spaces.
  {
    const int N_x        = 10000;
    const int N_y        = 20000;
    const int block_sz   = 256;
    const int n_blocks_x = (N_x + block_sz) / block_sz + 1;
    const int n_blocks_y = (N_y + block_sz) / block_sz + 1;

    using global_pol_y = RAJA::LoopPolicy<RAJA::cuda_global_thread_y>;
    using global_pol_x = RAJA::LoopPolicy<RAJA::cuda_global_thread_x>;

    RAJA::launch<launch_policy>
      (RAJA::LaunchParams(RAJA::Teams(n_blocks_x, n_blocks_y), RAJA::Threads(block_sz)),
       [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

	RAJA::loop<global_pol_y>(ctx, RAJA::RangeSegment(0, N_y), [&] (int gy) {
	    RAJA::loop<global_pol_x>(ctx, RAJA::RangeSegment(0, N_x), [&] (int gx) {

		//Do something

	      });
	  });

      });

  }

#else

  std::cout<<"Please compile with CUDA"<<std::endl;
#endif

  return 0;
}
