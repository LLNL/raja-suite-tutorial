#include <iostream>
#include <cstdlib>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  constexpr int N{1000};
  double* a{nullptr};
  double* b{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  a = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  b = static_cast<double*>(allocator.allocate(N*sizeof(double)));

  std::srand(4793);

  // Initialize data arrays to random positive and negative values
  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      double signfact = static_cast<double>(std::rand()/RAND_MAX);
      signfact = ( signfact < 0.5 ? -1.0 : 1.0 );

      a[i] = signfact * (i + 1.1)/(i + 1.12);
      b[i] = (i + 1.1)/(i + 1.12);
    }
  );

  // TODO: Change this dot variable to instead use a RAJA OpenMP parallel
  // reduction 
  double dot{0.0};

  // TODO: Calculate and output the dot product of a and b using a RAJA::forall
  RAJA::forall< RAJA::omp_parallel_for_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
    }
  );

  std::cout << "dot product is "<< dot << std::endl;

  allocator.deallocate(a);
  allocator.deallocate(b);

  return 0;
}
