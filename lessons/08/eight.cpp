#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/umpire.hpp"

int main()
{
  constexpr std::size_t SIZE{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  double* a;
  double* b;
  double* a_h;
  double* b_h;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("CUDA");
  auto host_allocator = rm.getAllocator("HOST");

  // TODO: create an allocator called "pool" using the QuickPool strategy

  a = pool.allocate(SIZE*sizeof(double));
  b = pool.allocate(SIZE*sizeof(double));
  a_h = host_allocator.allocate(SIZE*sizeof(double));
  b_h = host_allocator.allocate(SIZE*sizeof(double));

  RAJA::forall< RAJA::loop_exec >(
    RAJA::TypedRangeSegment<std::size_t>(0, SIZE), [=] (std::size_t i) {
      a_h[i] = 1.0;
      b_h[i] = 1.0;
    }
  );

  rm.copy(a, a_h);
  rm.copy(b, b_h);

  double dot{0.0};
  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, SIZE), 
    [=] RAJA_DEVICE (std::size_t i) { 
    cudot += a[i] * b[i]; 
  });    

  dot = cudot.get();

  allocator.deallocate(a);
  allocator.deallocate(b);
}
