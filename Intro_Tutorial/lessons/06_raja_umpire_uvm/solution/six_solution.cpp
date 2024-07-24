#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  constexpr int N{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  double* a{nullptr};
  double* b{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");

  a = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  b = static_cast<double*>(allocator.allocate(N*sizeof(double)));

  RAJA::forall< RAJA::cuda_exec<CUDA_BLOCK_SIZE> >(
    RAJA::TypedRangeSegment<int>(0, N), [=] RAJA_DEVICE (int i) {
      a[i] = 1.0;
      b[i] = 1.0;
    }
  );

  double dot{0.0};
  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::TypedRangeSegment<int>(0, N), 
    [=] RAJA_DEVICE (int i) { 
    cudot += a[i] * b[i]; 
  });    

  dot = cudot.get();

  std::cout << "dot = " << dot << std::endl;

  allocator.deallocate(a);
  allocator.deallocate(b);

  return 0;
}
