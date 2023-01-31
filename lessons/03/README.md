# Lesson 3

The fundamental concept for accessing memory through Umpire is the
umpire::Allocator. An umpire::Allocator is a C++ object that can be used to
allocate and deallocate memory, as well as query a pointer to get some extra
information about it.

All umpire::Allocator objects are created and managed by Umpire’s
umpire::ResourceManager. To create an allocator, first obtain a handle to the
ResourceManager, and then request the Allocator corresponding to the desired
memory resource using the `getAllocator` function.

Check the Umpire documenation for more details:
https://umpire.readthedocs.io/en/develop/sphinx/tutorial/allocators.html
 