#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "RAJA/RAJA.hpp"

#include "umpire/Umpire.hpp"

//TODO: uncomment this out in order to build!
#define COMPILE

int main()
{
  constexpr int N{1000000};
  double dx{1.0 / N};

  double* pi_h{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");

  pi_h = static_cast<double*>(host_allocator.allocate(1*sizeof(double)));  

  pi_h[0] = 0.0;

  // OpenMP kernel that approximates pi
#if defined(RAJA_ENABLE_OPENMP)
  {
    using EXEC_POL   = RAJA::omp_parallel_for_exec;
    using ATOMIC_POL = RAJA::omp_atomic;

    RAJA::forall<EXEC_POL>(RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
        double x = (double(i) + 0.5) * dx;
        RAJA::atomicAdd<ATOMIC_POL>( pi_h,  dx / (1.0 + x * x) );
    });
    pi_h[0] *= 4.0;

    std::cout << "OpenMP pi approximation " << " = "
              << std::setprecision(20) << pi_h[0] << std::endl;
  }
#endif

#if defined(COMPILE) && defined(RAJA_ENABLE_CUDA)
  // TODO: Implement pi approximation to run on CUDA device
  {
    constexpr std::size_t CUDA_BLOCK_SIZE{256};

    // TODO: Define CUDA execution policy, using the CUDA block size defined
    // above, and define the atomic policy 
    using EXEC_POL   = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
    using ATOMIC_POL = RAJA::cuda_atomic;

    pi_h[0] = 0.0;

    auto device_allocator = rm.getAllocator("DEVICE");

    double* pi_d{nullptr};

    // TODO: Allocate device data for 'pi_d' using the device allocator
    // defined above
    pi_d = static_cast<double*>(device_allocator.allocate(1*sizeof(double)));

    // TODO: Use the Umpire memset operation to initialize the data
    rm.memset(pi_d, 0);

    // TODO: Write a RAJA CUDA kernel to approximate pi
    RAJA::forall<EXEC_POL>(RAJA::TypedRangeSegment<int>(0, N), [=] __device__ (int i) {
        double x = (double(i) + 0.5) * dx;
        RAJA::atomicAdd<ATOMIC_POL>( pi_d,  dx / (1.0 + x * x) );
    });
  
    // TODO: Use the Umpire copy operation to copy the result in device memory
    // to the host array 'pi_h' so that the result can be printed below
    rm.copy(pi_h, pi_d, 1*sizeof(double));
    pi_h[0] *= 4.0;

    std::cout << "CUDA pi approximation " << " = "
              << std::setprecision(20) << pi_h[0] << std::endl;

    device_allocator.deallocate(pi_d);
  }
#endif // if defined(COMPILE) && defined(RAJA_ENABLE_CUDA)

  host_allocator.deallocate(pi_h);

  return 0;
}
