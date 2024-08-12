blt_add_executable(
  NAME 03_umpire_allocator
  SOURCES 03_umpire_allocator.cpp
  DEPENDS_ON RAJA umpire cuda)
