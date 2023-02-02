#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  constexpr std::size_t N{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  double* a{nullptr};
  double* b{nullptr};
  double* a_h{nullptr};
  double* b_h{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("DEVICE");
  auto host_allocator = rm.getAllocator("HOST");

  // TODO: create an allocator called "pool" using the QuickPool strategy

  a = pool.allocate(N*sizeof(double));
  b = pool.allocate(N*sizeof(double));
  a_h = host_allocator.allocate(N*sizeof(double));
  b_h = host_allocator.allocate(N*sizeof(double));

  RAJA::forall< RAJA::loop_exec >(
    RAJA::TypedRangeSegment<std::size_t>(0, N), [=] (std::size_t i) {
      a_h[i] = 1.0;
      b_h[i] = 1.0;
    }
  );

  rm.copy(a, a_h);
  rm.copy(b, b_h);

  double dot{0.0};
  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (std::size_t i) { 
    cudot += a[i] * b[i]; 
  });    

  dot = cudot.get();

  pool.deallocate(a);
  pool.deallocate(b);
  host_allocator.deallocate(a_h);
  host_allocator.deallocate(b_h);

  return 0;
}
