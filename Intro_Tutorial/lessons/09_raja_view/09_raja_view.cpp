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

  // TODO: Create a standard MxN RAJA::View called, "A", initialized with the "a" array.
  // TODO: Create a permuted MxN view with a right-oriented layout called, "R", initialized with the "result_right" array.
  constexpr int DIM = 2;
  auto L = RAJA::make_permuted_view<RAJA::layout_left>(result_left, M, N);

  // TODO: Fill in loop bounds that are appropriate for right-oriented layouts of Views A and R.
  for ( ??? )
  {
    for ( ??? )
    {
      // TODO: Initialize A and R views to their index values, e.g. index 0 should contain 0,
      // index 1 should contain 1, . . ., index 14 should contain 14. Note that both
      // A and R should print out the same sequence of values, due to the default
      // View for A being constructed with a right-oriented layout.
      A(row, col) = ???;
      R(row, col) = ???;
    }
  }

  // The L view will receive the same values as A and R. Note to achieve this,
  // the loop indexing is reversed from the previous initialization loops because L
  // is a left-oriented layout. The values assigned to L also reflect left-oriented
  // indexing arithmetic.
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

  // TODO: Look at the output and make sure each array prints the same ordering of values.
  // "a" and "result_right" should match "result_left".
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
