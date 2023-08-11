# Lesson 6

In this lesson, you will learn how to use Umpire's different memory resources to
allocate memory on a GPU. 

Each computer system will have a number of distinct places in which the system
will allow you to allocate memory. In Umpire’s world, these are memory
resources. A memory resource can correspond to a hardware resource, but can also
be used to identify memory with a particular characteristic, like “pinned”
memory in a GPU system.

Umpire creates predefined allocators for each of the available resources, and
they can be accessed using the `ResourceManager::getAllocator` method.

The predefined names can include:

- "HOST": CPU memory, like `malloc`.
- "DEVICE": device memory, and a "::<N>" suffix can be added to request memory on a specific device.
- "UM": unified memory that can be accessed by both the CPU and GPU.
- "PINNED": CPU memory that is pinned and will be accessible by the GPU.

In this example, you can use the "UM" resource so that the data can be accessed
by the CPU or GPU.

You will also find that we are adjusting the `RAJA::forall` to now work on the GPU.
In order for this to happen, we need a few extra things. First, we create a 
`CUDA_BLOCK_SIZE` variable to tell RAJA how big we want our CUDA blocks to be.
Since there are 32 threads in a warp, 256 tends to be a good value for a block size.
Other sizes will work too, such as 128 or 512. Typically, this value works well
at 256, but other values could also work well. This just depends on your GPU.

Additionally, the `RAJA::forall` needs the CUDA execution policy. More on GPU
execution policies can be found here: https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#gpu-policies-for-cuda-and-hip

The `cuda_exec` policy takes the cuda block size argument we created before
as a template parameter. Finally, as we are filling in the lambda portion of
the `RAJA::forall`, we need to specify where it will reside in GPU memory. 
This can be done directly or by using the `RAJA_DEVICE` macro. 

There are several `TODO` comments in the `six.cpp` exercise file where you 
can modify the code to work on a GPU. When you are done, build 
and run the example:

```
$ make six
$ ./bin/six
```

For more information on Umpire's resources, see our documentation:
https://umpire.readthedocs.io/en/develop/index.html

You can also read more about RAJA foralls and kernels here:
https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/add_vectors.html?highlight=RAJA_DEVICE#basic-loop-execution-vector-addition
and
https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/dot_product.html#raja-variants
