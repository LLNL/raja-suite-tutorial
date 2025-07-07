# Lesson 6 and 7

## Part 1: Lesson 6

For lesson 6, you will learn about Umpire's different memory resources and in
particular, those used to allocate memory on a GPU. 

Each computer system will have a number of distinct places in which the system
will allow you to allocate memory. In Umpire's world, these are memory
resources. A memory resource can correspond to a hardware resource, but can also
be used to identify memory with a particular characteristic, like `pinned`
memory in a GPU system.

Umpire creates predefined allocators for each of the available resources, and
they can be accessed using the `ResourceManager::getAllocator` method.

The predefined names can include:

- "HOST": CPU memory, like `malloc`.
- "DEVICE": device memory, and a "::<N>" suffix can be added to request memory on a specific device.
- "UM": unified memory that can be accessed by both the CPU and GPU.
- "PINNED": CPU memory that is pinned and will be accessible by the GPU.

Other memory resources include:

- "DEVICE_CONST": constant, read-only GPU memory
- "FILE": mmapped file memory that is accessible by the CPU.
- "SHARED": Includes POSIX shared memory which can be accessible by the CPU or GPU depending
on what your system accommodates and the MPI3 shared memory that is accessible on the CPU.
- "UNKNOWN": If an incorrect name is used or if the allocator was not set up correctly.

## Part 2: Lesson 7

For lesson 7, you will learn how to use Umpire's operations to copy data
between CPU and GPU memory in a portable way, using the memory resources you learned 
about in lesson 6.

In `07_raja_umpire_host_device.cpp`, we create an allocator for the GPU with:
```  
auto allocator = rm.getAllocator("DEVICE");
```

and a separate allocator on the CPU with:

```
  auto host_allocator = rm.getAllocator("HOST");
```

We will initialize the data on the CPU, but we want to do computations on
the GPU. Therefore, we have to take advantage of some Umpire "Operators".
In lesson 3, we learned how to use Umpire's `memset` operator. This lesson
builds on top of that to show other available operators.

Umpire provides a number of operations implemented as methods on the
`ResourceManager`. These typically take pointer and size arguments, but you do
not need to tell Umpire which Allocator each pointer came from. Umpire keeps
track of this and will call the appropriate underlying vendor function.

The copy method has the following signature:

```
void umpire::ResourceManager::copy (void* dst_ptr, void * src_ptr, std::size_t size = 0)	
```

*Note:* The destination is the first argument.

In the file `07_raja_umpire_host_device.cpp`, there is a `TODO` comment where you should insert two copy
calls to copy data from the CPU memory to the DEVICE memory.

You will also find that we are adjusting the `RAJA::forall` to now work on the GPU.
In order for this to happen, we need a few extra things. First, we create a 
`CUDA_BLOCK_SIZE` variable to tell RAJA how big we want our CUDA blocks to be.
Since there are 32 threads in a warp, 256 tends to be a good value for a block size.
Other sizes will work too, such as 128 or 512. This just depends on your GPU.

Additionally, the `RAJA::forall` needs the CUDA execution policy. More on GPU
execution policies can be found here: https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#gpu-policies-for-cuda-and-hip

The `cuda_exec` policy takes the cuda block size argument we created before
as a template parameter. Finally, as we are filling in the lambda portion of
the `RAJA::forall`, we need to specify where it will reside in GPU memory. 
This can be done directly or by using the `RAJA_DEVICE` macro. 

When you are done editing the file, compile and run it:

```
$ make 07_raja_umpire_host_device
$ ./bin/07_raja_umpire_host_device
```
