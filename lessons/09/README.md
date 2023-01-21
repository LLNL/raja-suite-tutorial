# Lesson One

In this lesson you will learn how to add RAJA and Umpire as dependencies to your application.

Like the previous lesson, we have a CMakeLists.txt file that will describe our
application using the `blt_add_executable` macro.

RAJA and Umpire are included in this project (look at tpl/CMakeLists.txt) and so
they exist as "targets" that we can tell CMake our application depends on. 

The `blt_add_executable` macro has another argument, `DEPENDS_ON`, that you can
use to list dependendencies.

! Add `RAJA` and `umpire` as dependiences using the DEPENDS_ON argument

Once you have added the dependencies, you can build and run the lesson as
before. As a reminder, open the VSCode terminal (Shift+^+`), and then move to
the build directory (`cd build`). You can then compile and run the lesson:

$ make two
$ ./two
Hello, world (with RAJA and Umpire)!

In the next lesson, we will start writing some code.