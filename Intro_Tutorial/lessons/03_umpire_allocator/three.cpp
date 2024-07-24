#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  double* data{nullptr};

  // TODO: allocate an array of 100 doubles using the HOST allocator

  std::cout << "Address of data: " << data << std::endl;

  // TODO: deallocate the array

  return 0;
}
