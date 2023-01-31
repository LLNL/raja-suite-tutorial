# Lesson 6

In this lesson you will learn how to use Umpire's different memory resources to
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

There is a TODO comment in six.cpp where you can modify the code to allocate GPU
memory. When you are done, build and run the example:

```
$ make six
$ ./bin/six
```

For more information on Umpire's resources, see our documentation:
https://umpire.readthedocs.io/en/develop/index.html