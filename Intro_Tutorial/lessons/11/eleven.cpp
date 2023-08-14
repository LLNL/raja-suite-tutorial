#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

//TODO: uncomment this out in order to build!
//#define COMPILE

int main()
{
#if defined(COMPILE)

  constexpr int N{10000};
  double* a{nullptr};
  double* b{nullptr};
  double* c{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  //TODO: Use a memory resource that allows you to access memory from both host
  //TODO: and device
  auto allocator = rm.getAllocator("????");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = static_cast<double *>(pool.allocate(N*N*sizeof(double)));
  b = static_cast<double *>(pool.allocate(N*N*sizeof(double)));
  c = static_cast<double *>(pool.allocate(N*N*sizeof(double)));

  constexpr int DIM = 2;

  RAJA::View<double, RAJA::Layout<DIM>> A(a, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> B(b, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> C(c, N, N);

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);

 // TODO: Create a EXEC_POL that uses CUDA


  //TODO: Uncomment out RAJA_DEVICE so that the kernel knows where to run
  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=] /*RAJA_DEVICE*/ (int col, int row) {
      A(row, col) = row;
      B(row, col) = col;
   });

  //TODO: Uncomment out RAJA_DEVICE so that the kernel knows where to run
  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=] /*RAJA_DEVICE*/ (int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += A(row, k) * B(k, col);
    }
    C(row, col) = dot;

  });

  pool.deallocate(a);
  pool.deallocate(b);
  pool.deallocate(c);

#endif
  
  return 0;
}
