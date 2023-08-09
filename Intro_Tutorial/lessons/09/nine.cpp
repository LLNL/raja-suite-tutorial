#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

//#define COMPILE

int main()
{
#if defined(COMPILE)

  constexpr int N{10000};
  double* a{nullptr};
  double* b{nullptr};
  double* c{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = static_cast<double *>(pool.allocate(N*N*sizeof(double)));
  b = static_cast<double *>(pool.allocate(N*N*sizeof(double)));
  c = static_cast<double *>(pool.allocate(N*N*sizeof(double)));

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);

  // TODO: Create a view for A, B, and C
  constexpr int DIM = 2;

  RAJA::forall<RAJA::loop_exec>( row_range, [=](int row) {
    RAJA::forall<RAJA::loop_exec>( col_range, [=](int col) {
      A(row, col) = row;
      B(row, col) = col;
    });
  });

  RAJA::forall<RAJA::loop_exec>( row_range, [=](int row) {
    RAJA::forall<RAJA::loop_exec>( col_range, [=](int col) {
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }
      C(row, col) = dot;
    });
  });

  pool.deallocate(a);
  pool.deallocate(b);
  pool.deallocate(c);

#endif

  return 0;
}
