# Lesson 3

In this lesson you will learn how to use Umpire to allocate memory. The file
three.cpp contains some TODO: comments where you can add code to allocate and
deallocate memory.

The fundamental concept for accessing memory through Umpire is the
`umpire::Allocator`. An `umpire::Allocator` is a C++ object that can be used to
allocate and deallocate memory, as well as query a pointer to get some extra
information about it.

All `umpire::Allocator` objects are created and managed by Umpire’s
`umpire::ResourceManager`. To create an allocator, first obtain a handle to the
ResourceManager, and then request the Allocator corresponding to the desired
memory resource using the `getAllocator` function:

```
auto& rm = umpire::ResourceManager::getInstance();
auto allocator = rm.getAllocator("HOST");
```

The Allocator class provides methods for allocating and deallocating memory. You
can view these methods in the Doxygen documentation here:
https://umpire.readthedocs.io/en/develop/doxygen/html/classumpire_1_1Allocator.html

```
void* memory = allocator.allocate(1024);
```

For more details, you can check out the Umpire documentation:
https://umpire.readthedocs.io/en/develop/sphinx/tutorial/allocators.html

Once you have made your changes, you can compile and run the lesson:

```
$ make three
$ ./bin/three
Address of data: 0x?????
```