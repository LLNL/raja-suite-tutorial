blt_add_executable(
  NAME 05_raja_reduce
  SOURCES 05_raja_reduce.cpp
  DEPENDS_ON RAJA umpire)

blt_add_executable(
  NAME 05_raja_reduce_solution
  SOURCES 05_raja_reduce_solution.cpp
  DEPENDS_ON RAJA umpire)
