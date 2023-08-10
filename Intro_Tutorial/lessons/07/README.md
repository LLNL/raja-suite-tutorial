# Lesson 7

In this lesson, you will learn how to use Umpire's operations to copy data
between CPU and GPU memory in a portable way.

In `seven.cpp`, we create an allocator for the GPU with:
```  
auto allocator = rm.getAllocator("DEVICE");
```

and a separete allocator on the CPU with:

```
  auto host_allocator = rm.getAllocator("HOST");
```

We will inialize the data on the CPU, but we want to do computations on
the GPU. Therefore, we have to take advantage of some Umpire "Operators".

Umpire provides a number of operations implemented as methods on the
`ResourceManager`. These typically take pointer and size arguments, but you do
not need to tell Umpire which Allocator each pointer came from. Umpire keeps
track of this and will call the appropriate underlying vendor function.

The copy method has the following signature:

```
void umpire::ResourceManager::copy (void* dst_ptr, void * src_ptr, std::size_t size = 0)	
```

*Note:* The destination is the first argument.

In the file `seven.cpp`, there is a `TODO` comment where you should insert two copy
calls to copy data from the CPU memory to the DEVICE memory.

When you are done editing the file, compile and run it:

```
$ make seven
$ ./bin/seven
```
