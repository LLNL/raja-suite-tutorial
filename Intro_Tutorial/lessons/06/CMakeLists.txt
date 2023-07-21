if (ENABLE_CUDA)
  blt_add_executable(
    NAME six
    SOURCES six.cpp
    DEPENDS_ON RAJA umpire cuda)
elseif (ENABLE_HIP)
  blt_add_executable(
    NAME six
    SOURCES six.cpp
    DEPENDS_ON RAJA umpire blt::hip)
endif()

