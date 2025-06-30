#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main()
{
  constexpr int N{10000};
  constexpr int M{7000};
  double* a{nullptr};
  double* a_t{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("UM");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = static_cast<double *>(pool.allocate(N*M*sizeof(double)));
  a_t = static_cast<double *>(pool.allocate(N*M*sizeof(double)));

  constexpr int DIM = 2;

  RAJA::View<double, RAJA::Layout<DIM>> A(a, N, M);
  RAJA::View<double, RAJA::Layout<DIM>> A_t(a_t, M, N);

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, M);

  using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernel<
          RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
	          RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=] RAJA_DEVICE (int col, int row) {
      A_t(col, row) = A(row, col);
   });

  pool.deallocate(a);
  pool.deallocate(a_t);

  return 0;
}
