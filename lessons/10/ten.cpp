#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/umpire.hpp"

int main()
{
  constexpr std::size_t N{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  constexpr std::size_t DIM{2};

  double* a;
  double* b;
  double* c;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = pool.allocate(SIZE*SIZE*sizeof(double));
  b = pool.allocate(SIZE*SIZE*sizeof(double));
  c = pool.allocate(SIZE*SIZE*sizeof(double));

  RAJA::View<double, RAJA::Layout<DIM>> A(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> B(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> C(C, N, N);

  // TODO: use RAJA::kernel to implement the nested loops
  // TODO: initialization loop

  // TODO: use RAJA::kernel to implement the nested loops
  // TODO: matrix multiply loop

  pool.deallocate(a);
  pool.deallocate(b);
  pool.deallocate(c);
}
