#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main()
{
  constexpr int N{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  constexpr std::size_t DIM{2};

  double* a{nullptr};
  double* b{nullptr};
  double* c{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

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
  using EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,    // row
        RAJA::statement::For<0, RAJA::loop_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
                         [=] (int col, int row) {
      A(row, col) = row;
      B(row, col) = col;
  });

  // TODO: use RAJA::kernel to implement the nested loops
  // TODO: matrix multiply loop
  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
                         [=] (int col, int row) {
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }
      C(row, col) = dot;
  });


  pool.deallocate(a);
  pool.deallocate(b);
  pool.deallocate(c);

  return 0;
}
