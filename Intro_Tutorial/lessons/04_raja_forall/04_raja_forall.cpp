#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"

#include "umpire/Umpire.hpp"

int main()
{
  double* data{nullptr};
  double* data1{nullptr};

  auto timer = RAJA::Timer();

  constexpr int N = 5000000;

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  data = static_cast<double*>(allocator.allocate(N*sizeof(double)));
  data1 = static_cast<double*>(allocator.allocate(N*sizeof(double)));

  std::cout << "Address of data: " << data << std::endl;
  std::cout << "Address of data1: " << data1 << std::endl;

  // Sequential kernel that sets each element of array 'data' to its index
  timer.start();
  RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      data[i] = i;
  });
  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  std::cout << "\nSequential loop exec time = " << elapsed << std::endl;
  std::cout << "data[50] = " << data[50] << std::endl;
  std::cout << "data[100] = " << data[100] << std::endl;
  std::cout << "data[1000] = " << data[1000] << std::endl;
  std::cout << "data[5000] = " << data[5000] << std::endl;

  timer.reset();

  timer.start();
  // TODO: write a parallel RAJA forall loop using OpenMP to set each element of the 
  // array 'data1' to its index
  timer.stop();

  elapsed = timer.elapsed();

  std::cout << "\nOpenMP loop exec time = " << elapsed << std::endl;
  std::cout << "data1[50] = " << data1[50] << std::endl;
  std::cout << "data1[100] = " << data1[100] << std::endl;
  std::cout << "data1[1000] = " << data1[1000] << std::endl;
  std::cout << "data1[5000] = " << data1[5000] << std::endl;

  allocator.deallocate(data);
  allocator.deallocate(data1);
  return 0;
}
