#include "RAJA/RAJA.hpp"

// In a serial GPU programing model kernels are executed 
// in a sequential order based on the order in which the 
// the kernels are being launched. 
// 
// GPU programming models such as CUDA/HIP and SYCL
// have the capability of performing device operations
// concurrently. The RAJA portability suite exposes 
// concurrent kernel execution through the use of the 
// raja::resources.
//
// A raja::resources corresponds to device stream in
// which we may guarantee that device operations will
// be executed in sequential order. Different streams,
// however; may operate concurrently. 
//

//
// RAJA::resources by default is configured to not be
// the device's default stream. Historicaly the default
// stream serves as a synchronizing stream. No other 
// operations can begin until all issued operations on 
// the default stream are completed.

// In modern versions of CUDA the behavior of the default
// stream can be changed to be non-synchronizing.



int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv))
{

#if defined(RAJA_ENABLE_CUDA)


  constexpr int N = 10;
  constexpr int M = 1000000;

  //Master resource to orchestrate between memory transfers
  RAJA::resources::Cuda def_cuda_res{RAJA::resources::Cuda::get_default()};
  RAJA::resources::Host def_host_res{RAJA::resources::Host::get_default()};
  int* d_array = def_cuda_res.allocate<int>(N*M);
  int* h_array = def_host_res.allocate<int>(N*M);

  RAJA::RangeSegment one_range(0, 1);
  RAJA::RangeSegment m_range(0, M);

  using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<true>>;

  using outer_pol_x = RAJA::LoopPolicy<RAJA::cuda_block_x_loop>;

  using inner_pol_x = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;


  for(int i=0; i<N; i++) {
    
    
    //Each new resource will intialize a new stream
    RAJA::resources::Cuda res_cuda;

    //Kernels will be executed asynchrously
    //A handle e can be used to synchronize between stream activity
      RAJA::resources::Event e =
        RAJA::launch<launch_policy>(res_cuda,
        RAJA::LaunchParams(RAJA::Teams(64),
			   RAJA::Threads(1)), "RAJA resource example",
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx)  {

       RAJA::loop<outer_pol_x>(ctx, m_range, [&] (int j) {
         RAJA::loop<inner_pol_x>(ctx, one_range, [&] (int k) {

           d_array[i*M + j] = i * M + j;

           });
         });

      });

      //perform synchronization between different streams
      def_cuda_res.wait_for(&e);
  }

  //Master resource to perform the memory copy
  //All other streams have been synchronized with respect to def_cuda_res
  def_cuda_res.memcpy(h_array, d_array, sizeof(int) * N * M);

  int ec_count = 0;
  RAJA::forall<RAJA::seq_exec>( RAJA::RangeSegment(0, N*M),
    [=, &ec_count](int i){
      if (h_array[i] != i) ec_count++;
    }
  );

  std::cout << "    Result -- ";
  if (ec_count > 0)
    std::cout << "FAIL : error count = " << ec_count << "\n";
  else
    std::cout << "PASS!\n";


#else
  std::cout<<"Please compile with CUDA"<<std::endl;
#endif

  return 0;
}
