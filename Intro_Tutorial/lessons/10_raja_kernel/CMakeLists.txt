blt_add_executable(
  NAME 10_raja_kernel
  SOURCES 10_raja_kernel.cpp
  DEPENDS_ON cuda RAJA umpire)
