# Lesson 8

In this lesson, you will learn to create a memory pool using Umpire.

Frequently allocating and deallocating memory can be quite costly, especially when you are making large allocations or allocating on different memory resources. 
Memory pools are a more efficient way to allocate large amounts of memory, especially when dealing with HPC environments.

Additionally, Umpire provides allocation strategies that can be used to customize how data is obtained from the system.
In this lesson, we will learn about one such strategy called `QuickPool`. 

The `QuickPool` strategy describes a certain type of pooling algorithm provided in the Umpire API. 
As its name suggests, `QuickPool` has been shown to be performant for many use cases. 

Umpire also provides other types of pooling strategies such as `DynamicPoolList` and `FixedPool`. 
You can visit the documentation to learn more: https://umpire.readthedocs.io/en/develop/index.html 

To create a new memory pool allocator using the `QuickPool` strategy, we can use the `ResourceManager`:
```
  umpire::Allocator pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool_name", my_allocator);
```

This newly created `pool` is an `umpire::Allocator` using the `QuickPool` strategy. As you can see above, we can use the `ResourceManager::makeAllocator` function to create the pool allocator. We just need to pass 
in: (1) the name we would like the pool to have, and (2) the allocator we previously created with the `ResourceManager` (see line 17 in the
file `eight.cpp`).

There are other arguments that could be passed to the pool constructor if needed. See the documentation page for more: https://umpire.readthedocs.io/en/develop/doxygen/html/index.html

When you have created your QuickPool allocator, compile and run the code:
```
$ make eight
$ ./bin/eight
```
