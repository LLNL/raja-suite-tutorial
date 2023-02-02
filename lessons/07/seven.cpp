#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  constexpr int N{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  double* a{nullptr};
  double* b{nullptr};
  double* a_h{nullptr};
  double* b_h{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("CUDA");
  auto host_allocator = rm.getAllocator("HOST");

  a = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  b = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  a_h = static_cast<double*>(host_allocator.allocate(N*sizeof(double)));
  b_h = static_cast<double*>(host_allocator.allocate(N*sizeof(double)));

  RAJA::forall< RAJA::loop_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      a_h[i] = 1.0;
      b_h[i] = 1.0;
    }
  );

  // TODO: copy data from a_h to a, and b_h to b

  double dot{0.0};
  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::TypedRangeSegment<int>(0, SIZE), 
    [=] RAJA_DEVICE (int i) { 
    cudot += a[i] * b[i]; 
  });    

  dot = cudot.get();

  allocator.deallocate(a);
  allocator.deallocate(b);
  host_allocator.deallocate(a_h);
  host_allocator.deallocate(b_h);

  return 0;
}
