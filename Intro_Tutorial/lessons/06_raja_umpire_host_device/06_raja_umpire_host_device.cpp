#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

//TODO: uncomment this out in order to build!
//#define COMPILE

int main()
{
#if defined(COMPILE)

  constexpr int N{1000};
  //TODO: Set up a block size value
  constexpr std::size_t CUDA_BLOCK_SIZE{????};
  double* a{nullptr};
  double* b{nullptr};
  double* a_h{nullptr};
  double* b_h{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  // TODO: create 2 allocators, one with device memory and one with host memory
  auto allocator = rm.getAllocator("??");
  auto host_allocator = rm.getAllocator("??");

  a = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  b = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  a_h = static_cast<double*>(host_allocator.allocate(N*sizeof(double)));
  b_h = static_cast<double*>(host_allocator.allocate(N*sizeof(double)));

  //TODO: fill in the forall statement with the sequential execution policy.
  RAJA::forall< ????? >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      a_h[i] = 1.0;
      b_h[i] = 1.0;
    }
  );

  // TODO: copy data from a_h to a, and b_h to b (i.e. from the host to the device vars)

  double dot{0.0};

  //TODO: create a RAJA::ReduceSum with cuda_reduce called "cudot" for the GPU

  //TODO: fill in the forall statement with the CUDA execution policy
  //TODO: and its block size argument. Then be sure to use RAJA_DEVICE
  RAJA::forall<?????<????>>(RAJA::TypedRangeSegment<int>(0, N), 
    [=] ???? (int i) { 
    cudot += a[i] * b[i]; 
  });    

  dot = cudot.get();

  std::cout << "dot = " << dot << std::endl;

  allocator.deallocate(a);
  allocator.deallocate(b);
  host_allocator.deallocate(a_h);
  host_allocator.deallocate(b_h);

#endif
  return 0;
}
