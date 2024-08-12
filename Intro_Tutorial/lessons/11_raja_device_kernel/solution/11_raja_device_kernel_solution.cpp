#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main()
{
  constexpr int N{10000};
  double* a{nullptr};
  double* b{nullptr};
  double* c{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("UM");
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

 // TODO: convert EXEC_POL to use CUDA
  using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixed<256,
          RAJA::statement::For<1, RAJA::cuda_global_size_y_direct<16>,
	    RAJA::statement::For<0, RAJA::cuda_global_size_x_direct<16>,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=] RAJA_DEVICE (int col, int row) {
      A(row, col) = row;
      B(row, col) = col;
   });

  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=] RAJA_DEVICE (int col, int row) {

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
