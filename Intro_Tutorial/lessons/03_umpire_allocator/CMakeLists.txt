blt_add_executable(
  NAME 03_umpire_allocator
  SOURCES 03_umpire_allocator.cpp
  DEPENDS_ON RAJA umpire cuda)

blt_add_executable(
  NAME 03_umpire_allocator_solution
  SOURCES 03_umpire_allocator_solution.cpp
  DEPENDS_ON RAJA umpire cuda)
