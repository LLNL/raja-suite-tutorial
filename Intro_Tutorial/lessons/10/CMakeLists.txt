blt_add_executable(
  NAME ten
  SOURCES ten.cpp
  DEPENDS_ON cuda RAJA umpire)
