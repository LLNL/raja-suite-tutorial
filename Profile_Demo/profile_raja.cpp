#include <stdexcept>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

#include "caliper-plugin.cpp"

// Basic profiling:
//  CALI_CONFIG=runtime-report ./bin/profile_raja 1024
//
// Show cuda activity
//  CALI_CONFIG=cuda-activity-report,show_kernels ./bin/profile_raja 1024
//
// Generate nsys profile
// CALI_SERVICES_ENABLE=nvtx,cuda nsys profile -o my_profile ./bin/profile_raja 1024

//Uncomment for policy selection

#define DIRECT_POLICY
//#define LOOP_POLICY
//#define GLOBAL_POLICY

constexpr int cuda_threads = 256;
constexpr int launch_max_threads = 256; //TODO add to cuda_launch_t
constexpr bool async = false;
using forall_pol = RAJA::cuda_exec<cuda_threads,async>;
using launch_pol = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async/*,launch_max_threads*/>>;

void init(double *A, double *B, double *C, int m, int n) {

  RAJA::forall<forall_pol>(RAJA::RangeSegment(0, n * n),
                           RAJA::Name("init"),
     [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
       A[i] = 1.0;
       B[i] = 1.0;
       C[i] = 0.0;
     });
}

void matrix_add(const double *A, const double *B, double *C, int m, int n) {

  RAJA::forall<forall_pol>
    (RAJA::RangeSegment(0, m * n), RAJA::Name("matrix_add"), [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
        C[i] = A[i] + B[i];
    });

}

void matrix_scalar_mult(const double *A, double *B, double scalar, int m, int n) {

  RAJA::forall<forall_pol>
    (RAJA::RangeSegment(0, m * n), RAJA::Name("matrix_scalar_mult"), [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
        B[i] = scalar * A[i];
  });
}

void matrix_multiply(const double *A, const double *B, double *C, int m, int n, int p) {

  // A: m x n, B: n x p, C: m x p
  auto v_A = RAJA::make_permuted_view<RAJA::layout_right>(A, m, n);
  auto v_B = RAJA::make_permuted_view<RAJA::layout_right>(B, n, p);
  auto v_C = RAJA::make_permuted_view<RAJA::layout_right>(C, m, p);

#if defined(DIRECT_POLICY)
  const int threads = p;
  const int teams = m;

  RAJA::LaunchParams params{RAJA::Teams(teams), RAJA::Threads(threads)};

  //Rows: if (int i = blockIdx.x; i < m)
  using loop1_pol = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

  //Cols: if (int j = threadIdx.x; j < p)
  using loop0_pol = RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;
#endif

#if defined(LOOP_POLICY)
  const int threads = 256;
  const int teams = m;

  RAJA::LaunchParams params{RAJA::Teams(teams), RAJA::Threads(threads)};

  //Rows: if (int i = blockIdx.x; i < m; i += gridDim.x)
  using loop1_pol = RAJA::LoopPolicy<RAJA::cuda_block_x_loop>;

  //Cols: if (int j = threadIdx.x; j < p; j += blockDim.x)
  using loop0_pol = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;
#endif

#if defined(GLOBAL_POLICY)
  constexpr int threads = 16;
  const int teams_x = (n - 1)/threads + 1;
  const int teams_y = (m - 1)/threads + 1;

  RAJA::LaunchParams params{RAJA::Teams(teams_x, teams_y), RAJA::Threads(threads, threads)};

  //Rows: int i = threadIdx.y + blockIdx.y * blockDim.y
  using loop1_pol = RAJA::LoopPolicy<RAJA::cuda_global_size_y_direct<threads>>;

  //Cols: int j = threadIdx.x + blockIdx.x * blockDim.x
  using loop0_pol = RAJA::LoopPolicy<RAJA::cuda_global_size_x_direct<threads>>;
#endif

  RAJA::launch<launch_pol>
  (params, RAJA::Name("matrix_multiply"),
   [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<loop1_pol>(ctx, RAJA::RangeSegment(0, m), [&] (int i) {
          RAJA::loop<loop0_pol> (ctx, RAJA::RangeSegment(0, p), [&] (int j) {

            double dot =0.0;
            for (int k = 0; k < n; k++) {
              dot += v_A(i, k) * v_B(k, j);
            }
            v_C(i, j) = dot;
           });
         });
    });
}


bool check_matrix_multiply(const double *C, const int n)
{

  bool pass = true;
  auto v_C = RAJA::make_permuted_view<RAJA::layout_right>(C, n, n);

  for(int r=0; r<n; ++r) {
    for(int c=0; c<n; ++c) {

      if(v_C(r, c) != n) {
        pass = false;
      }
    }
  }
  return pass;
}

int main(int argc, char* argv[])
{

  if(argc != 2) {
    throw std::runtime_error("usage ./main N -- where N is matrix size (N x N )");
  }

  int n = std::atoi(argv[1]);
  std::cout<<"Using matrix size "<<n<<" x "<<n<<std::endl;

  double* h_A{nullptr};
  double* h_B{nullptr};
  double* h_C{nullptr};

  double* d_A{nullptr};
  double* d_B{nullptr};
  double* d_C{nullptr};

  //Use host and device memory
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto device_allocator = rm.getAllocator("DEVICE");

  h_A = static_cast<double*>(host_allocator.allocate(n*n*sizeof(double)));
  h_B = static_cast<double*>(host_allocator.allocate(n*n*sizeof(double)));
  h_C = static_cast<double*>(host_allocator.allocate(n*n*sizeof(double)));

  d_A = static_cast<double*>(device_allocator.allocate(n*n*sizeof(double)));
  d_B = static_cast<double*>(device_allocator.allocate(n*n*sizeof(double)));
  d_C = static_cast<double*>(device_allocator.allocate(n*n*sizeof(double)));

  init(d_A, d_B, d_C, n, n);

  matrix_add(d_A, d_B, d_C, n, n);

  matrix_scalar_mult(d_A, d_C, 2.0, n, n);

  matrix_multiply(d_A, d_B, d_C, n, n, n);

  rm.copy(h_C, d_C, n*n*sizeof(double));

  bool pass = check_matrix_multiply(h_C, n);

  if(!pass) {
    throw std::runtime_error("matrix_multiply did not pass");
  }

  std::cout<<"Matrix multiply passed"<<std::endl;

  host_allocator.deallocate(h_A);
  host_allocator.deallocate(h_B);
  host_allocator.deallocate(h_C);

  device_allocator.deallocate(d_A);
  device_allocator.deallocate(d_B);
  device_allocator.deallocate(d_C);

  return 0;
}
