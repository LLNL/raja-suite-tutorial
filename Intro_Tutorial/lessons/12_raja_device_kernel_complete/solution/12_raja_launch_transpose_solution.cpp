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

  using EXEC_POL =
      RAJA::cuda_launch_t<false>;
  using teams_x = RAJA::cuda_block_x_direct;
  using threads_x = RAJA::cuda_thread_x_loop;

  RAJA::launch<EXEC_POL>(RAJA::ExecPlace::DEVICE,
    RAJA::LaunchParams(RAJA::Teams(N), RAJA::Threads(M)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
      RAJA::loop<teams_x>(ctx, col_range, [&] (int col) {
        RAJA::loop<threads_x>(ctx, row_range, [&] (int row) {
          A_t(col, row) = A(row, col);
        });
      });
    });

  pool.deallocate(a);
  pool.deallocate(a_t);

  return 0;
}