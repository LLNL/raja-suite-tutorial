#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

template<typename U, typename V>
void check_solution(U &A, V &A_t, const int M, const int N);

int main()
{

  constexpr int N{10000};
  constexpr int M{7000};
  double* h_a{nullptr};
  double* h_a_t{nullptr};
  double* d_a{nullptr};
  double* d_a_t{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto device_allocator = rm.getAllocator("DEVICE");
  auto host_allocator = rm.getAllocator("HOST");

  d_a = static_cast<double *>(device_allocator.allocate(N*M*sizeof(double)));
  d_a_t = static_cast<double *>(device_allocator.allocate(N*M*sizeof(double)));
  h_a = static_cast<double *>(host_allocator.allocate(N*M*sizeof(double)));
  h_a_t = static_cast<double *>(host_allocator.allocate(N*M*sizeof(double)));

  auto h_A   = RAJA::make_permuted_view<RAJA::layout_right>(h_a, M, N);
  auto h_A_t   = RAJA::make_permuted_view<RAJA::layout_right>(h_a_t, N, M);

  // Intialize data
  for(int row = 0; row < M; ++row) {
    for(int col = 0; col < N; ++col) {
      h_A(row, col) = col + N * row;
    }
  }

  rm.copy(d_a, h_a, N*M*sizeof(double));
  rm.copy(d_a_t, h_a_t, N*M*sizeof(double));

  auto d_A   = RAJA::make_permuted_view<RAJA::layout_right>(d_a, M, N);
  auto d_A_t = RAJA::make_permuted_view<RAJA::layout_right>(d_a_t, N, M);

  constexpr int team_size = 16;
  const int teams_x   = (M - 1) / team_size + 1;
  const int teams_y   = (N - 1) / team_size + 1;

  const bool async = false;
  using EXEC_POL =
    RAJA::LaunchPolicy<RAJA::cuda_launch_t<async>>;
  using outer_loop = RAJA::LoopPolicy<RAJA::cuda_global_size_y_direct<team_size>>;
  using inner_loop = RAJA::LoopPolicy<RAJA::cuda_global_size_x_direct<team_size>>;

  RAJA::launch<EXEC_POL>(
    RAJA::LaunchParams(RAJA::Teams(teams_x, teams_y), RAJA::Threads(team_size, team_size)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<inner_loop>(ctx, RAJA::TypedRangeSegment<int>(0,M), [&] (int row) {
          RAJA::loop<outer_loop>(ctx, RAJA::TypedRangeSegment<int>(0,N), [&] (int col) {
          d_A_t(col, row) = d_A(row, col);
        });
      });
    });

  rm.copy(h_a, d_a, N*M*sizeof(double));
  rm.copy(h_a_t, d_a_t, N*M*sizeof(double));

  check_solution(h_A, h_A_t, M, N);

  device_allocator.deallocate(d_a);
  device_allocator.deallocate(d_a_t);
  host_allocator.deallocate(h_a);
  host_allocator.deallocate(h_a_t);

  return 0;
}

template<typename U, typename V>
void check_solution(U &A, V &A_t, const int M, const int N)
{
  bool pass = true;

  for(int row = 0; row < M; ++row) {
    for(int col = 0; col < N; ++col) {
      if(A(row, col) != A_t(col, row)) {
        pass = false;
      }
    }
  }

  if(pass) {
    std::cout<<"SUCCESS! Matrix transpose passed"<<std::endl;
  }else{
    std::cout<<"Error! Matrix transpose did not pass"<<std::endl;
  }

}
