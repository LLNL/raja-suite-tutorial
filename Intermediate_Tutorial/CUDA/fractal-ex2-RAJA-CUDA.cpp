//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <malloc.h>
#include <sys/time.h>

#include "RAJA/RAJA.hpp"
#include "../../tpl/writeBMP.hpp"

#define xMin 0.74395
#define xMax 0.74973
#define yMin 0.11321
#define yMax 0.11899

/* TODO: create a variable called "THREADS" to be used when calling the kernel*/
#define THREADS 512 

int main(int argc, char *argv[])
{
  double dx, dy;
  int width, maxdepth;
  struct timeval start, end;
  writebmp wbmp;

  /* check command line */
  if(argc != 3) {fprintf(stderr, "usage: exe <width> <depth>\n"); exit(-1);}
  width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "edge_length must be at least 10\n"); exit(-1);}
  maxdepth = atoi(argv[2]);
  if (maxdepth < 10) {fprintf(stderr, "max_depth must be at least 10\n"); exit(-1);}

  dx = (xMax - xMin) / width;
  dy = (yMax - yMin) / width;

  printf("computing %d by %d fractal with a maximum depth of %d\n", width, width, maxdepth);

  /* TODO: Create the "cnt" array to store the pixels and allocate space for it on CPU using pinned memory */
  unsigned char *cnt;
  cudaHostAlloc((void**)&cnt, (width * width * sizeof(unsigned char)), cudaHostAllocDefault);

  /* TODO: Create the "d_cnt" array to store pixels on the GPU and allocate space for it on the GPU */
  unsigned char *d_cnt;
  cudaMalloc((void**)&d_cnt, width * width * sizeof(unsigned char));

  /* TODO: Set up a RAJA::KernelPolicy. The Policy should describe a cuda kernel with one outer loop 
   * and one inner loop. Only the inner for loop will be calculating pixels. 
   */
  using KERNEL_POLICY = RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<1, RAJA::cuda_block_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
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
    //Remember, RAJA takes care of finding the global thread ID, so just index into the image like normal
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
      d_cnt[row * width + col] = depth & 255; //Remember to index the image like normal
  });
  gettimeofday(&end, NULL); //By the time we exit the RAJA::Kernel, host and device are synchronized for us.

  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  /* TODO: In order to create a bmp image, we need to copy the completed fractal to the Host memory space */
  cudaMemcpyAsync(cnt, d_cnt, width * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  /* verify result by writing it to a file */
  if (width <= 2048) {
    wbmp.WriteBMP(width, width, cnt, "fractal.bmp");
  }

  /* TODO: Free the memory we allocated. */
  cudaFreeHost(cnt);
  cudaFree(d_cnt);
  return 0;
}