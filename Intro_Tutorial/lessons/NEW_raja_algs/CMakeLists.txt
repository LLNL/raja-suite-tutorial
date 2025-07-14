blt_add_executable(
  NAME NEW_raja_atomic
  SOURCES NEW_raja_atomic.cpp
  DEPENDS_ON RAJA umpire cuda)

##blt_add_executable(
##  NAME NEW_raja_scan
##  SOURCES NEW_raja_scan.cpp
##  DEPENDS_ON RAJA umpire cuda)
