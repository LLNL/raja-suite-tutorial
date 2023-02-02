#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  constexpr std::size_t N{10000};
  double* a;
  double* b;
  double* c;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = pool.allocate(N*N*sizeof(double));
  b = pool.allocate(N*N*sizeof(double));
  c = pool.allocate(N*N*sizeof(double));

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);

  // TODO: Create a view for A, B, and C

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
        dot += Aview(row, k) * Bview(k, col);
      }
      C(row, col) = dot;
    });
  });

  pool.deallocate(a);
  pool.deallocate(b);
  pool.deallocate(c);

  return 0;
}
