#include <iostream>

int main(int argc, char *argv[])
{


  auto& rm = umpire::ResourceManager::getInstance();
  unsigned char *cnt{nullptr};
  auto allocator = rm.getAllocator("UM");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("qpool", allocator);
  cnt = static_cast<unsigned char*>(pool.allocate(width * width * sizeof(unsigned char)));

  using device_launch = RAJA::cuda_launch_t<false>;
  using launch_policy = RAJA::LaunchPolicy<device_launch>;

  //Example 1. Global Indexing: 
  //GPU programming models such as CUDA and HIP follow a thread/block(team) programming model 
  //in which a predefined compute grid 
  {
    const int N_x        = 10000;
    const int N_y        = 20000;
    const int block_sz   = 256;
    const int n_blocks_x = (N_x + block_sz) / block_sz + 1;
    const int n_blocks_y = (N_y + block_sz) / block_sz + 1;

    using loop_pol_x = RAJA::LoopPolicy<RAJA::cuda_global_x>;

    RAJA::launch<device_launch>
      (RAJA::LaunchParams(RAJA::Teams(n_blocks_x, n_blocks_y), RAJA::Threads(block_sz)),
       [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

	RAJA::loop<loop_pol_y>(ctx, RAJA::RangeSegment(0, N_y), [&] (int gy) {
	    RAJA::loop<loop_pol_x>(ctx, RAJA::RangeSegment(0, N_x), [&] (int gx) {

		//populate 
		
		
	      });
	  });


      });

  }









  //Iteration Space:
  {
  const int n_blocks = 50000;
  const int block_sz = 64;

  RAJA::launch<launch_policy>
    ( RAJA::LaunchParams(RAJA::Teams(n_blocks),
			RAJA::Threads(block_sz)),
     [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {


      RAJA::loop<col_loop>(ctx, RAJA::RangeSegment(0, width), [&] (int col) {

	});

    });
  }
  






  return 0;
}
