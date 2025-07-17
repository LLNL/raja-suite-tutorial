#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  double* data{nullptr};

  // TODO: allocate an array of 100 doubles using the HOST allocator

  // TODO: use the resource manager to memset your array to 0

  // TODO: uncomment this print statement
  //std::cout << "Allocated " << (100 * sizeof(double)) << " bytes and set to "
  //          << data[0] << " using the " << allocator.getName() << " allocator."
  //          << std::endl;

  // TODO: deallocate the array

  return 0;
}
