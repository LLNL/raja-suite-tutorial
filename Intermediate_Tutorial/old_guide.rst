================
Fractal Tutorial
================

This tutorial includes several implementations of a Mandelbrot set Fractal code.
The code originated from Dr. Martin Burtscher of the Efficient Computing Lab at
Texas State University. You can find more here: https://userweb.cs.txstate.edu/~burtscher/research.html

In the "extras" directories, there are a few other RAJA implementations such 
as OpenMP and even a native CUDA implementation just for comparison. You can reference
these implementation to study the differences in implementation and runtime comparison.
However, anything beyond that is outside the scope of this tutorial.

To start, let's build the tutorial within a newly created, empty `build` directory located
in the `raja-suite-tutorial` repo:: 

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

To verify your results in each lesson, you can look at the resulting .bmp file output. If you
have completed everything correctly, you will see a complete image of the fractal.
Currently, there is an `if` statement that makes sure the `writeBMP` function
is only called for smaller fractal runs (of width <= 2048). You can edit this `if` statement, but be careful because trying
to write a .bmp file that is too large will take a very long time.

Continue on to the first lesson located in the `00-BASE` directory.
