#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/umpire.hpp"

int main()
{
  constexpr SIZE{10000};
  double* a;
  double* b;
  double* c;

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  a = allocator.allocate(SIZE*sizeof(double));
  b = allocator.allocate(SIZE*sizeof(double));

  RAJA::forall< RAJA::loop_exec >(
    RAJA::TypedRangeSegment<std::size_t>(0, SIZE), [=] (std::size_t i) {
      a[i] = 1.0;
      b[i] = 1.0;
    }
  );

  // TODO: use a reduction to calculate the dotproduct of a and b

  allocator.deallocate(a);
  allocator.deallocate(b);
}
