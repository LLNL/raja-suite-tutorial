#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  double* data{nullptr};

  // TODO: allocate an array of 100 doubles using the HOST allocator
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");
  void* memory = allocator.allocate(100*sizeof(double));

  std::cout << "Address of data: " << data << std::endl;

  // TODO: deallocate the array
  allocator.deallocate(memory);

  return 0;
}
