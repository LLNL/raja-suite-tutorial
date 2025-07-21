blt_add_executable(
  NAME 04_raja_forall
  SOURCES 04_raja_forall.cpp
  DEPENDS_ON RAJA umpire)

blt_add_executable(
  NAME 04_raja_forall_solution
  SOURCES 04_raja_forall_solution.cpp
  DEPENDS_ON RAJA umpire)
