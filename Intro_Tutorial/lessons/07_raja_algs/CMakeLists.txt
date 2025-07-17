blt_add_executable(
  NAME 07_raja_atomic
  SOURCES 07_raja_atomic.cpp
  DEPENDS_ON RAJA umpire cuda)

blt_add_executable(
  NAME 07_raja_scan
  SOURCES 07_raja_scan.cpp
  DEPENDS_ON RAJA umpire cuda)
