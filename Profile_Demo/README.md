# Basic RAJA profiling with Caliper

In this example, we explore profiling RAJA kernels using the Caliper library developed at LLNL. 
Below are example build commands you can use to configure Caliper and RAJA for profiling on NVIDIA GPUs.

Building Caliper on an NVIDIA platform:
``cmake -DCMAKE_INSTALL_PREFIX=${caliper_path} -DWITH_NVTX=ON -DWITH_CUPTI=ON ../``

Building RAJA:
``cmake -DENABLE_CUDA=ON -DRAJA_ENABLE_RUNTIME_PLUGINS=ON -DRAJA_ENABLE_CALIPER=ON -Dcaliper_DIR=${caliper_path}/build/share/cmake/caliper -DCMAKE_CUDA_FLAGS="--expt-extended-lambda" -Dcaliper_DIR=${caliper_path} ../ && make profile_raja -j``

Once the suite is built, you can invoke the following command to profile a set of basic linear algebra kernels:

``CALI_CONFIG=runtime-report ./bin/profile_raja 1024``

This example provides three different kernel policies, allowing users to observe runtime performance differences between the kernels. 
To switch between them, uncomment the desired variable at the top of the file.

For more information on Caliper we refer the reader to the following pages:

- [RAJA-Caliper Quick Start Documentation](https://raja.readthedocs.io/en/develop/sphinx/user_guide/profiling_with_caliper.html)
- [Caliper GitHub](https://github.com/LLNL/Caliper)
- [Caliper Documentation](https://software.llnl.gov/Caliper/)