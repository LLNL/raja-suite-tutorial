In the following lessons you will compare implementations of a fractal generating kernel
using the kernel and launch frameworks.

The first three examples the RAJA kernel method is used while in the forth example
RAJA launch is used. RAJA kernel requires recompilation when changing backend dispatch.
RAJA launch supports run time selection between a host and device backend. 

As before the exercises have COMPILE macro guards, to compile the code uncomment the 
COMPILE define on top of the file. 
