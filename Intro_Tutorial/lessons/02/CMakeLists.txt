  # TODO: Add the RAJA, umpire and cuda targets as DEPENDS_ON arguments
blt_add_executable(
  NAME two
  SOURCES two.cpp
  DEPENDS_ON cuda RAJA umpire)
