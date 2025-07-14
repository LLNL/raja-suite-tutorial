#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main()
{
  constexpr int M{3};
  constexpr int N{5};
  double* a{nullptr};
  double* result_right{nullptr};
  double* result_left{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = static_cast<double *>(pool.allocate(M*N*sizeof(double)));
  result_right = static_cast<double *>(pool.allocate(M*N*sizeof(double)));
  result_left = static_cast<double *>(pool.allocate(M*N*sizeof(double)));

  // TODO: Create a standard MxN RAJA::View called, "A". 
  // TODO: Create a permuted MxN view with a right-oriented layout called, "R".
  constexpr int DIM = 2;
  RAJA::View<double, RAJA::Layout<DIM>> A(a, M, N);
  auto R = RAJA::make_permuted_view<RAJA::layout_right>(result_right, M, N);
  auto L = RAJA::make_permuted_view<RAJA::layout_left>(result_left, M, N);

  // The A and R views are initialized to their index values. Note that both
  // A and R should be the same, due to the default View being constructed
  // with a right-oriented layout.
  for ( int row = 0; row < M; ++row )
  {
    for ( int col = 0; col < N; ++col )
    {
      A(row, col) = row * N + col;
      R(row, col) = row * N + col;
      L(row, col) = 0.0;
    }
  }

  // TODO: Finish the loops and assignment of values to L such that
  // the data ordering in the result_left array matches that of A and R.
  for ( int col = 0; col < N; ++col )
  {
    for ( int row = 0; row < M; ++row )
    {
      L(row, col) = col * M + row;
    }
  }

  auto printArrayAsMatrix = [&](double * array)
  {
    for ( int ii = 0; ii < M*N; ++ii )
    {
      printf("%f ", array[ii]);
      if ( ((ii+1) % N == 0) )
      {
        printf("\n");
      }
    }
  };

  printf("\na array under View A:\n");
  printArrayAsMatrix( a );

  printf("\nresult_right array under View R:\n");
  printArrayAsMatrix( result_right );

  printf("\nresult_left array under View L:\n");
  printArrayAsMatrix( result_left );

  pool.deallocate(a);
  pool.deallocate(result_right);
  pool.deallocate(result_left);

  return 0;
}
