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

  // TODO: write a RAJA forall loop to initialize each element of a to the value
  // of the index

  std::cout << "data[50] = " << data[50] << std::endl;

  allocator.deallocate(data);
}
