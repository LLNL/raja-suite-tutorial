# Lesson 1

In this lesson you will learn how to use BLT and CMake to build an executable.

RAJA and Umpire use BLT and CMake as their build systems, and we recommend them
for other applications, like this tutorial! CMake uses information in a set of
CMakeLists.txt to generate files to build your project. In this case, we will be
using the `make` program to actual compile everything.

BLT provides a set of CMake macros that make it easy to write CMake code for HPC
applications targetting multiple hardware architectures.

We won't give you a full CMake/BLT tutorial here, just enough to get things moving.

Our top-level CMakeLists file describes the project, sets up some options, and
then calls `add_subdirectory` so that CMake looks for more CMakeLists.txt files.

In this directory, we have a CMakeLists.txt file that will describe our
application. We use the `blt_add_executable` macro to do this.

The macro takes two (or more) arguments, and the two we care about at the moment
are `NAME` where you provide the executable name, and `SOURCES` where you list
all the source code files that make up your application:

  blt_add_executable(
    NAME one
    SOURCES one.cpp)

For now, we have filled these out for you, but in later lessons you will need to
make some edits yourself.

For a full tutorial on BLT, please see: https://llnl-blt.readthedocs.io/en/develop/tutorial/index.html

## Building the Lessons 

We have already run CMake for you in this container to generate the make-based
build system. So now you can compile and run the first lesson.

First, open the VSCode terminal (Shift+^+`), and then move to the
build directory:

$ cd build

Compiling your project in a different directory to the source code is a best
practice when using CMake.  Once you are in the build directory, you can use the
`make` command to compile the executable:

$ make one

You will see some output as the code is compiled. You can then run the
executable:

$ ./bin/one
Hello, world!

In the next lesson, we will show you how to add RAJA and Umpire as dependencies
to the application.

