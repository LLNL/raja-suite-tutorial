# Lesson 1: BLT and CMake

In this lesson, you will learn how to use BLT and CMake to configure a 
software project to build. RAJA and Umpire use BLT and CMake as their build
systems, and so does this tutorial. 

[CMake](https://cmake.org/) is a common tool for building C++ projects and is
used throughout the world. It uses information in `CMakeLists.txt` files located in a software project to generate configuration files to build a code project 
on a particular system. Then, a utility like `make` can be used to compile
the code.

[BLT](https://github.com/LLNL/blt) provides a foundation of CMake macros and 
other tools that simplify the process of Building, Linking, and Testing high
performance computing (HPC) applications. In particular, BLT establishes best
practices for using CMake. 

The goal with this lesson is not to give you a full CMake/BLT tutorial. We
want to give you enough information to help get you started configuring and 
building the code in this tutorial.

Our top-level [CMakeLists.txt file](https://github.com/LLNL/raja-suite-tutorial/blob/main/CMakeLists.txt)  describes this project, sets some options, 
and then calls `add_subdirectory`, telling CMake to look in sub-directories for
more CMakeLists.txt files.

In this lesson directory, we have a CMakeLists.txt file that describes our
application. We use the `blt_add_executable` macro to do this.

The macro takes two (or more) arguments, and the two most important
are `NAME` where you provide the name of the executable to be generated, and
`SOURCES` where you list all the source code files to compile to generate the
executable:

```
blt_add_executable(
    NAME 01_blt_cmake
    SOURCES 01_blt_cmake.cpp)
```

For more information on BLT, please refer to the [BLT User Guide and Tutorial](https://llnl-blt.readthedocs.io/en/develop/tutorial/index.html).

## Building the Lessons 

We have already run CMake for you in the container used for this tutorial
to generate the make-based build system. So you are ready to compile and run
the first lesson.

First, open the VSCode terminal (Shift + ^ + `), and then move to the
build directory:

```
$ cd build
```

Compiling your project in a different directory than the source code is a best
practice when using CMake. Once you are in the build directory, you can use the
`make` command to compile the executable:

```
$ make 01_blt_cmake
```

You will see some output as the code is compiled. You can then run the
executable:

```
$ ./bin/01_blt_cmake
Hello, world!
```

In the next lesson, we will show you how to add RAJA and Umpire as dependencies
to an application.
