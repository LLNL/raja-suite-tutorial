blt_add_executable(
  NAME 09_raja_view
  SOURCES 09_raja_view.cpp
  DEPENDS_ON cuda RAJA umpire)

blt_add_executable(
  NAME 09_raja_view_solution
  SOURCES 09_raja_view_solution.cpp
  DEPENDS_ON cuda RAJA umpire)
