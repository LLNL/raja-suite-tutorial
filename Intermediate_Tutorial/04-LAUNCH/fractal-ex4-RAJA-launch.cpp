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

//#define COMPILE

int main(int argc, char *argv[])
{
#if defined(COMPILE)

  double dx, dy;
  int width;
  const int maxdepth = 256;
  struct timeval start, end;
  writebmp wbmp;

  /* check command line */
  if(argc != 3) {fprintf(stderr, "usage: exe host <width> or exe device <width> \n"); exit(-1);}

  std::string exec_space = argv[1];
  if(!(exec_space.compare("host") == 0 || exec_space.compare("device") == 0 )){
    RAJA_ABORT_OR_THROW("usage: exe host <width> or exe device <width>");
    return 0;
  }

  RAJA::ExecPlace select_cpu_or_gpu;
  if(exec_space.compare("host") == 0)
    { select_cpu_or_gpu = RAJA::ExecPlace::HOST; printf("Running RAJA-Launch fractals example on the host \n"); }
  if(exec_space.compare("device") == 0)
    { select_cpu_or_gpu = RAJA::ExecPlace::DEVICE; printf("Running RAJA-Launch fractals example on the device \n"); }

  width = atoi(argv[2]);
  if (width < 10) {fprintf(stderr, "edge_length must be at least 10\n"); exit(-1);}

  dx = (xMax - xMin) / width;
  dy = (yMax - yMin) / width;

  printf("computing %d by %d fractal with a maximum depth of %d\n", width, width, maxdepth);

  //TODO: Create an Umpire QuickPool allocator with Unified Memory that will hold the
  // pixels of the fractal image.
  auto& rm = umpire::ResourceManager::getInstance();
  unsigned char *cnt{nullptr};
  auto allocator = rm.getAllocator("UM");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("qpool", allocator);
  cnt = static_cast<unsigned char*>(pool.allocate(width * width * sizeof(unsigned char)));

  //TODO: Create a RAJA launch policy for the host and device

  using host_launch =

#if defined(RAJA_ENABLE_CUDA)
  using device_launch =
#elif defined(RAJA_ENABLE_HIP)
  using device_launch =
#endif

  using launch_policy = RAJA::LaunchPolicy<
    host_launch
#if defined(RAJA_GPU_ACTIVE)
    ,device_launch
#endif
    >;

  //RAJA loop policies take a pair of policies enabling run time selection of

  using col_loop = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::cuda_global_thread_x>;

  using row_loop = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::cuda_global_thread_y>;

  /* start time */
  gettimeofday(&start, NULL);

  constexpr int block_sz = 16;
  int n_blocks = (width + block_sz - 1) / block_sz + 1;

  RAJA::launch<launch_policy>
    (select_cpu_or_gpu, RAJA::LaunchParams(RAJA::Teams(n_blocks, n_blocks),
                                           RAJA::Threads(block_sz, block_sz)),
     [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<col_loop>(ctx, RAJA::RangeSegment(0, width), [&] (int col) {
          RAJA::loop<row_loop>(ctx, RAJA::RangeSegment(0, width), [&] (int row) {

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
        });

    });

  /* end time */
  gettimeofday(&end, NULL);
  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  /* verify result by writing it to a file */
  if (width <= 2048) {
    wbmp.WriteBMP(width, width, cnt, "fractal.bmp");
  }

  //TODO: Use the Umpire pooled allocator to deallocate the memory.

#endif
  return 0;
}
