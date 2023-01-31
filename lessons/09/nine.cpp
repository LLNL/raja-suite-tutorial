#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/umpire.hpp"

int main()
{
  constexpr std::size_t SIZE{10000};
  double* a;
  double* b;
  double* c;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = pool.allocate(SIZE*SIZE*sizeof(double));
  b = pool.allocate(SIZE*SIZE*sizeof(double));
  c = pool.allocate(SIZE*SIZE*sizeof(double));

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

  allocator.deallocate(a);
  allocator.deallocate(b);
}