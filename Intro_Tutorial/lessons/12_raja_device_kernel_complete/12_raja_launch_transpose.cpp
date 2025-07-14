#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main()
{
#if defined (COMPILE)
  constexpr int N{10000};
  constexpr int M{7000};
  double* a{nullptr};
  double* a_t{nullptr};
  double* a_h{nullptr};
  double* a_t_h{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("DEVICE");
  auto host_allocator = rm.getAllocator("HOST");

  a = static_cast<double *>(allocator.allocate(N*M*sizeof(double)));
  a_t = static_cast<double *>(allocator.allocate(N*M*sizeof(double)));
  a_h = static_cast<double *>(host_allocator.allocate(N*M*sizeof(double)));
  a_t_h = static_cast<double *>(host_allocator.allocate(N*M*sizeof(double)));

  rm.copy(a, a_h, N*M*sizeof(double));
  rm.copy(a_t, a_t_h, N*M*sizeof(double));

  constexpr int DIM = 2;

  RAJA::View<double, RAJA::Layout<DIM>> A(a, N, M);
  RAJA::View<double, RAJA::Layout<DIM>> A_t(a_t, M, N);

  using EXEC_POL =
      RAJA::cuda_launch_t<false>;
  using outer_loop = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
  using inner_loop = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;

  RAJA::launch<EXEC_POL>(
    // TODO: Which numbers of Teams and Threads should be used in this launch?
    RAJA::LaunchParams(RAJA::Teams(???), RAJA::Threads(???)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
      RAJA::loop<outer_loop>(ctx, RAJA::TypedRangeSegment<int>(0,N), [&] (int col) {
        RAJA::loop<inner_loop>(ctx, RAJA::TypedRangeSegment<int>(0,M), [&] (int row) {
          // TODO: What indexing should be used to fill in the transposed View?
          A_t(???, ???) = A(row, col);
        });
      });
    });

  allocator.deallocate(a);
  allocator.deallocate(a_t);
  host_allocator.deallocate(a_h);
  host_allocator.deallocate(a_t_h);
#endif

  return 0;
}
