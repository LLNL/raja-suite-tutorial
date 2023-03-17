================
Fractal Tutorial
================

This tutorial includes several implementations of a Mandelbrot set Fractal code.
The code originated from Dr. Martin Burtscher of the Efficient Computing Lab at
Texas State University. You can find more here: https://userweb.cs.txstate.edu/~burtscher/research.html

The tutorial first starts with a RAJA loop-exec policy which is very similar to a serial 
RAJA loop implementation of the fractal code. From there, we edit the loop-exec version
to instead use RAJA-CUDA RAJA-HIP execution policies. Plus, I have added a few "extras"
which includes a few other RAJA implementations such as OpenMP and even a native CUDA
implementation just for comparison.

TODO: The final lessons include a more complex fractal implementation that includes
RAJA-TEAMS.

To start, let's build the tutorial within the build directory of the RAJA repo:: 

        module load cuda/11.2.0
        module load cmake/3.20.2
        module load gcc/8.3.1
        cmake -DENABLE_CUDA=On -DENABLE_OPENMP=Off -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-11.2.0/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-11.2.0 -DBLT_CXX_STD=c++14 -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_EXERCISES=On -DRAJA_ENABLE_OPENMP=Off -DCMAKE_CUDA_FLAGS=--extended-lambda -DCUDA_ARCH=sm_70 ../

.. note::
        I am building this code on LC's lassen machine. If these build instructions don't work for you, you can refer to the
        build documentation from RAJA's ReadTheDocs or use one of the provided build scripts.

Now, we can build the RAJA loop-exec implementation with `./bin/fractal 1024`. The first argument
is the width of the fractal (1024). It may be interesting to see how the fractal changes with 
different width values.

Be sure to study the loop-exec implementation of the fractal before continuing. It is important to note:
 * Read-only, write-only, and read-write variables used in the main computation
 * The main data structure that holds the values of the fractal pixels
 * Any data dependencies, if any, throughout the computation of the pixels

TODO: Add more info on what RAJA's loop-exec does and why it's a good first step for learning.

Next, we will parallelize this using RAJA-CUDA.

