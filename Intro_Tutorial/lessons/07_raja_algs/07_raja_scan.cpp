#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "RAJA/RAJA.hpp"

#include "umpire/Umpire.hpp"

//TODO: uncomment this out in order to build!
//#define COMPILE

int main()
{
  constexpr int N{10};

  int in[N] = {8,-1,2,9,10,3,4,1,6,7};
  int is_out[N] = {};
  int es_out[N] = {};

  // OpenMP scan kernels
#if defined(RAJA_ENABLE_OPENMP)
  {
    using EXEC_POL   = RAJA::omp_parallel_for_exec;

    RAJA::inclusive_scan<EXEC_POL>( RAJA::make_span(in, N),
                                    RAJA::make_span(is_out, N) );

    std::cout << "Output (inclusive): ";
    for (int i = 0; i < N; ++i) {
       std::cout << is_out[i] << "  ";
    }
    std::cout << std::endl;

    //////////////

    RAJA::exclusive_scan<EXEC_POL>( RAJA::make_span(in, N),
                                    RAJA::make_span(es_out, N) );

    std::cout << "Output (exclusive): ";
    for (int i = 0; i < N; ++i) {
      std::cout << es_out[i] << "  ";
    }
    std::cout << std::endl;

    //////////////

    RAJA::inclusive_scan_inplace<EXEC_POL>( RAJA::make_span(in, N) );

    std::cout << "Output (inclusive in-place): ";
    for (int i = 0; i < N; ++i) {
      std::cout << in[i] << "  ";
    }
    std::cout << std::endl;

  }
#endif

#if defined(COMPILE) && defined(RAJA_ENABLE_CUDA)
  // TODO: Implement RAJA scan to run on CUDA device
  {
    constexpr int M{20};
    
    int* array_h{nullptr};
    int* array_d{nullptr};

    auto& rm = umpire::ResourceManager::getInstance();

    auto host_allocator = rm.getAllocator("HOST");
    array_h = static_cast<int*>(host_allocator.allocate(M*sizeof(int)));
    for (int i = 0; i < M; ++i) {
      array_h[i] = i - 1;
    }

    // TODO: Create a device memory alloctor, allocate array 'array_d'
    //       on the device, and initialize the device array by using the
    //       Umpire copy operation to copy the values from 'array_h'.
    auto device_allocator = ???;
    array_d = ???;
    rm.copy( ??? , ??? , ??? );

    // TODO: Write a RAJA operation to do an exclusive in-place scan on a 
    //       GPU using CUDA using the array 'array_d' and a maximum operation
    constexpr std::size_t CUDA_BLOCK_SIZE{128};
    RAJA::exclusive_scan_inplace< ??? >(
      ???, RAJA::operators::maximum<int>{});

    // TODO: Use the Umpire copy operation to copy the result in device memory
    // to the host array 'array_h' so that the result can be printed below
    rm.copy( ??? , ??? , ??? );

    std::cout << "Output (exclusive (CUDA) in-place): ";
    for (int i = 0; i < M; ++i) {
      std::cout << array_h[i] << "  ";
    }
    std::cout << std::endl;

  }
#endif // if defined(COMPILE) && defined(RAJA_ENABLE_CUDA)

  return 0;
}
