#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  constexpr int N{10000};
  double* a{nullptr};
  double* b{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  a = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  b = static_cast<double*>(allocator.allocate(N*sizeof(double)));

  RAJA::ReduceSum< RAJA::seq_reduce, double > dot(0.0);

  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      a[i] = 1.0;
      b[i] = 1.0;
      dot += a[i] * b[i];
    }
  );

  // TODO: use a reduction to calculate and output the dotproduct of a and b

  std::cout << "dot product is "<< dot << std::endl;

  allocator.deallocate(a);
  allocator.deallocate(b);

  return 0;
}
