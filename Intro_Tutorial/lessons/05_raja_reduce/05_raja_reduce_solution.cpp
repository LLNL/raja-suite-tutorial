#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "RAJA/RAJA.hpp"

//TODO: uncomment this out in order to build!
#define COMPILE

int main()
{
  constexpr int N{1000000};

  constexpr double dx{1.0 / N}; 

  // Sequential kernel that approximates pi 
  {
    RAJA::ReduceSum<RAJA::seq_reduce, double> pi(0.0);
  
    RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
        double x = (double(i) + 0.5) * dx;
        pi += dx / (1.0 + x * x);
    });
    double tpi = 4.0 * pi.get(); 

    std::cout << "Sequential pi approximation " << " = " 
              << std::setprecision(20) << tpi << std::endl;
  }

#if defined(COMPILE)
  // OpenMP kernel that approximates pi
  {
    // TODO: write a parallel RAJA forall loop using OpenMP to approximate pi.
    // First, will have to define create a RAJA::ReduceSum object that will
    // work in an OpenMP kernel.
    RAJA::ReduceSum<RAJA::omp_reduce, double> pi(0.0);

    RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
        double x = (double(i) + 0.5) * dx;
        pi += dx / (1.0 + x * x);
    });
    double tpi = 4.0 * pi.get();

    std::cout << "OpenMP pi approximation " << " = "
              << std::setprecision(20) << tpi << std::endl;
  }
#endif  // if defined(COMPILE)

  return 0;
}
