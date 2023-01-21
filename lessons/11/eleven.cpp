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

  auto allocator = rm.getAllocator("UM");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = pool.allocate(SIZE*SIZE*sizeof(double));
  b = pool.allocate(SIZE*SIZE*sizeof(double));
  c = pool.allocate(SIZE*SIZE*sizeof(double));

 // TODO: convert EXEC_POL to use CUDA
 using EXEC_POL =
    RAJA::KernelPolicy<
          // ??
    >;

  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=] (int col, int row) {
      A(row, col) = row;
      B(row, col) = col;
    }

  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=](int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += A(row, k) * B(k, col);
    }
    C(row, col) = dot;

  });

  pool.deallocate(a);
  pool.deallocate(b);
  pool.deallocate(c);
}
