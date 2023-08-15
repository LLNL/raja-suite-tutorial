#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

//#define COMPILE

int main()
{

#if defined(COMPILE)

  constexpr int N{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  constexpr std::size_t DIM{2};

  double* a{nullptr};
  double* b{nullptr};
  double* c{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("myPOOL", allocator);

  a = static_cast<double *>(pool.allocate(N*N*sizeof(double)));
  b = static_cast<double *>(pool.allocate(N*N*sizeof(double)));
  c = static_cast<double *>(pool.allocate(N*N*sizeof(double)));

  RAJA::View<double, RAJA::Layout<DIM>> A(a, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> B(b, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> C(c, N, N);

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);

  // TODO: use RAJA::kernel to implement the nested loops
  // TODO: initialization loop



  // TODO: use RAJA::kernel to implement the nested loops
  // TODO: matrix multiply loop


  pool.deallocate(a);
  pool.deallocate(b);
  pool.deallocate(c);

#endif 

  return 0;
}
