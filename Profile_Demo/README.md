# Basic RAJA profiling with Caliper

Building Caliper:
cmake -DCMAKE_INSTALL_PREFIX=${caliper_path} -DWITH_NVTX=ON -DWITH_CUPTI=ON ../

Building RAJA:
cmake -DENABLE_CUDA=ON -DRAJA_ENABLE_RUNTIME_PLUGINS=ON -DRAJA_ENABLE_CALIPER=ON -DCMAKE_CUDA_FLAGS="--expt-extended-lambda" -Dcaliper_DIR=${caliper_path} ../ && make profile_raja -j

TODO: Add to readme
