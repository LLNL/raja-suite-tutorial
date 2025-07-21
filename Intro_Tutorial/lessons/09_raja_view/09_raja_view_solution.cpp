#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

// TODO: Uncomment this in order to build!
#define COMPILE

// Method to print arrays associated with the Views in the lesson
void printArrayAsMatrix( double * array, int row, int col )
{
  for ( int ii = 0; ii < row * col; ++ii )
  {
    std::cout << array[ii] << " ";
    if ( ((ii+1) % col == 0) )
    {
      std::cout << std::endl;
    }
  }
}

int main()
{
#if defined(COMPILE)
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

  constexpr int DIM = 2;

  // TODO: Create a standard MxN RAJA::View called "A", initialized with the
  // "a" array.
  RAJA::View<double, RAJA::Layout<DIM>> A(a, M, N);

  // A left-oriented layout view initialized with the "result_left" array
  auto L = RAJA::make_permuted_view<RAJA::layout_left>(result_left, M, N);

  // TODO: Create a permuted MxN view with a right-oriented layout called "R",
  // initialized with the "result_right" array.
  auto R = RAJA::make_permuted_view<RAJA::layout_right>(result_right, M, N);

  // Note that Views created by RAJA::make_permuted_view know the unit stride
  // index at compile time, which prevents unnecessary index arithmetic.

  // TODO: Fill in loop bounds that are appropriate for right-oriented layout
  // Views A and R.
  for ( int row = 0; row < M; ++row )
  {
    for ( int col = 0; col < N; ++col )
    {
      // TODO: Initialize A and R views to their index values, e.g. index 0
      // should contain 0, index 1 should contain 1, . . ., index 14 should
      // contain 14.
      //
      // Note that both A and R should print out the same sequence of values
      // in the calls to 'printArrayAsMatrix' below.
      A(row, col) = row * N + col;
      R(row, col) = row * N + col;
    }
  }

  // The L view will receive the same values as A and R. To achieve this,
  // the loop indexing is reversed from the previous initialization loops
  // because L is a left-oriented layout. The values assigned to L also
  // reflect left-oriented indexing arithmetic.
  for ( int col = 0; col < N; ++col )
  {
    for ( int row = 0; row < M; ++row )
    {
      L(row, col) = col * M + row;
    }
  }

  // TODO: Run the code and review the output from the following method calls
  // to make sure each array prints the same ordering of values.
  // "a" and "result_right" should match "result_left".
  std::cout << "a array under View A:" << std::endl;
  printArrayAsMatrix( a, M, N );

  std::cout << "result_right array under View R:" << std::endl;
  printArrayAsMatrix( result_right, M, N );

  std::cout << "result_left array under View L:" << std::endl;
  printArrayAsMatrix( result_left, M, N );

  pool.deallocate(a);
  pool.deallocate(result_right);
  pool.deallocate(result_left);
#endif

  return 0;
}
