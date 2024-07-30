//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <malloc.h>
#include <sys/time.h>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "../../tpl/writeBMP.hpp"

#define xMin 0.74395
#define xMax 0.74973
#define yMin 0.11321
#define yMax 0.11899

#define THREADS 256

int main(int argc, char *argv[])
{
  double dx, dy;
  int width;
  const int maxdepth = 256;
  struct timeval start, end;
  writebmp wbmp;

  /* check command line */
  if(argc != 2) {fprintf(stderr, "usage: exe <width>\n"); exit(-1);}
  width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "edge_length must be at least 10\n"); exit(-1);}

  dx = (xMax - xMin) / width;
  dy = (yMax - yMin) / width;

  printf("computing %d by %d fractal with a maximum depth of %d\n", width, width, maxdepth);

  //TODO: Create an Umpire QuickPool allocator with unified memory that will hold the
  //pixels of the fractal image.
  auto& rm = umpire::ResourceManager::getInstance();
  unsigned char *cnt{nullptr};
  auto allocator = rm.getAllocator("UM");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("qpool", allocator);
  cnt = static_cast<unsigned char*>(pool.allocate(width * width * sizeof(unsigned char)));

  /* TODO: Set up a RAJA::KernelPolicy. The Policy should describe a hip kernel with one outer loop 
   * and one inner loop. Only the inner for loop will be calculating pixels. 
   */
  using KERNEL_POLICY = RAJA::KernelPolicyFixed<
    RAJA::statement::HipKernel<THREADS,
      RAJA::statement::For<1, RAJA::hip_global_size_y_direct<16>,
        RAJA::statement::For<0, RAJA::hip_global_size_x_direct<16>,
          RAJA::statement::Lambda<0>
        >
      > 
    >
  >;
  
  /* compute fractal */
  gettimeofday(&start, NULL);
  /* TODO: Add a RAJA::Kernel which takes the KERNEL_POLICY you just created above.
   * It should take range segments that go the same range as our for-loops from before.
   * The iterators inside the kernel body will describe the row and col of the image.
   */
  RAJA::kernel<KERNEL_POLICY>(
        RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, width),
                         RAJA::TypedRangeSegment<int>(0, width)),
        [=] RAJA_DEVICE (int row, int col) {
    double x2, y2, x, y, cx, cy;
    int depth;

      /* compute fractal */
      cy = yMin + row * dy; //compute row #
      cx = xMin + col * dx; //compute column #
      x = -cx;
      y = -cy;
      depth = maxdepth;
      do {
        x2 = x * x;
        y2 = y * y;
        y = 2 * x * y - cy;
        x = x2 - y2 - cx;
        depth--;
      } while ((depth > 0) && ((x2 + y2) <= 5.0));
      cnt[row * width + col] = depth & 255; //Remember to index the image like normal
  });
  gettimeofday(&end, NULL); //By the time we exit the RAJA::Kernel, host and device are synchronized for us.

  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  /* verify result by writing it to a file */
  if (width <= 2048) {
    wbmp.WriteBMP(width, width, cnt, "fractal.bmp");
  }

  /* TODO: Free the memory we allocated. */
  pool.deallocate(cnt);
  return 0;
}
