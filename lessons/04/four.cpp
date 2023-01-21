#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/umpire.hpp"

int main()
{
  double* data;

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  a = allocator.allocate(100*sizeof(double));

  std::cout << "Address of data: " << data << std::endl;

  // write a RAJA forall loop to initialize a
  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      data[i] = 1.0*i;
    }
  );

  std::cout << "data[50] = " << data[50] << std::endl;

  allocator.deallocate(data);
}
