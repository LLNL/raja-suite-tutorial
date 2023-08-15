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

int main(int argc, char *argv[])
{
  double dx, dy;
  int width;
  const int maxdepth = 256;
  struct timeval start, end;
  writebmp wbmp;

  /* check command line */
  if(argc != 2) {fprintf(stderr, "usage: exe <width> \n"); exit(-1);}
  width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "edge_length must be at least 10\n"); exit(-1);}

  dx = (xMax - xMin) / width;
  dy = (yMax - yMin) / width;

  printf("computing %d by %d fractal with a maximum depth of %d\n", width, width, maxdepth);

  //TODO: Create an Umpire QuickPool allocator with host memory that will hold the
  //pixels of the fractal image.
  auto& rm = umpire::ResourceManager::getInstance();
  unsigned char *cnt{nullptr};
  auto allocator = rm.getAllocator("HOST");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("qpool", allocator);
  cnt = static_cast<unsigned char*>(pool.allocate(width * width * sizeof(unsigned char)));

  //TODO: Create a RAJA Kernel Policy which uses the seq_exec policy. We want to start
  //with a normal serial nested seq first before continuing onward.
  using KERNEL_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;

  /* start time */
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

  //TODO: Use the Umpire pooled allocator to deallocate the memory.
  pool.deallocate(cnt);
  return 0;
}
