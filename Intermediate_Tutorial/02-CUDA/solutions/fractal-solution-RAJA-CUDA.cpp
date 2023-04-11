//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <malloc.h>
#include <sys/time.h>

#include "RAJA/RAJA.hpp"
#include "../../../tpl/writeBMP.hpp"

#define xMin 0.74395
#define xMax 0.74973
#define yMin 0.11321
#define yMax 0.11899

#define THREADS 512 

int main(int argc, char *argv[])
{
  double dx, dy;
  int width; 
  int maxdepth = 256;
  unsigned char *cnt;
  struct timeval start, end;
  writebmp wbmp;


  /* check command line */
  if(argc != 2) {fprintf(stderr, "usage: exe <width>\n"); exit(-1);}
  width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "edge_length must be at least 10\n"); exit(-1);}

  unsigned char *d_cnt;
 
  dx = (xMax - xMin) / width;
  dy = (yMax - yMin) / width;

  printf("computing %d by %d fractal with a maximum depth of %d\n", width, width, maxdepth);

  cudaHostAlloc((void**)&cnt, (width * width * sizeof(unsigned char)), cudaHostAllocDefault);

  /* allocate space on GPU */
  cudaMalloc((void**)&d_cnt, width * width * sizeof(unsigned char));

  using KERNEL_POLICY = RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<1, RAJA::cuda_block_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
          RAJA::statement::Lambda<0>
        >
      > 
    >
  >;

  gettimeofday(&start, NULL);
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
      d_cnt[row * width + col] = depth & 255;
  });
  gettimeofday(&end, NULL);

  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  cudaMemcpyAsync(cnt, d_cnt, width * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  /* verify result by writing it to a file */
  if (width <= 2048) {
    wbmp.WriteBMP(width, width, cnt, "fractal.bmp");
  }

  cudaFreeHost(cnt);
  cudaFree(d_cnt);
  return 0;
}
