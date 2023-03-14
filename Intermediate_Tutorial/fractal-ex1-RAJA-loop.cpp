#include <malloc.h>
#include <sys/time.h>

#include "RAJA/RAJA.hpp"
#include "../tpl/writeBMP.hpp"

#define xMin 0.74395
#define xMax 0.74973
#define yMin 0.11321
#define yMax 0.11899

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

  unsigned char *cnt = (unsigned char*)malloc(width * width * sizeof(unsigned char));

  using KERNEL_POLICY = RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::Lambda<0>
        >
      > 
  >;

  gettimeofday(&start, NULL);
  RAJA::kernel<KERNEL_POLICY>(
        RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, width),
                         RAJA::TypedRangeSegment<int>(0, width)),
        [=] (int row, int col) {
    double x2, y2, x, y, cx, cy;
    int depth;

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
    cnt[row * width + col] = depth & 255;
  });
						
  /* end time */
  gettimeofday(&end, NULL);
  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  /* verify result by writing it to a file */
  if (width <= 2048) {
    wbmp.WriteBMP(width, width, cnt, "fractal.bmp");
  }

  free(cnt);
  return 0;
}