#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"

int main()
{
  double* data{nullptr};

  constexpr int N = 100;

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  data = static_cast<double*>(allocator.allocate(N*sizeof(double)));

  std::cout << "Address of data: " << data << std::endl;

  // TODO: write a RAJA forall loop to set each element of the array 'data' to 
  // the value of the loop index

  RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      data[i] = i;
  });

  std::cout << "data[50] = " << data[50] << std::endl;

  allocator.deallocate(data);
  return 0;
}
