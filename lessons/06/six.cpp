#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/umpire.hpp"

int main()
{
  constexpr std::size_t SIZE{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  double* a;
  double* b;
  double* c;

  auto& rm = umpire::ResourceManager::getInstance();
  // TODO: allocate with CUDA
  auto allocator = rm.getAllocator("??");

  a = allocator.allocate(SIZE*sizeof(double));
  b = allocator.allocate(SIZE*sizeof(double));

  RAJA::forall< RAJA::cuda_exec<CUDA_BLOCK_SIZE> >(
    RAJA::TypedRangeSegment<std::size_t>(0, SIZE), [=] (std::size_t i) {
      a[i] = 1.0;
      b[i] = 1.0;
    }
  );

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
