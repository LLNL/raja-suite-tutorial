blt_add_executable(
  NAME 09_raja_view_makepermuted
  SOURCES 09_raja_view_makepermuted.cpp
  DEPENDS_ON cuda RAJA umpire)
