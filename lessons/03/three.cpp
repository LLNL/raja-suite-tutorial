#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/umpire.hpp"

int main()
{
  double* a{nullptr};

  // TODO: allocate an array of 100 doubles using the HOST allocator

  std::cout << "Address of a: " << a << std::endl;


  // TODO: deallocate a

}
