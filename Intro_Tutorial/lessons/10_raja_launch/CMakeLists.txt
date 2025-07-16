if (ENABLE_CUDA)
  blt_add_executable(
    NAME 12_raja_device_kernel_complete
    SOURCES 12_raja_device_kernel_complete.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()

if (ENABLE_CUDA)
  blt_add_executable(
    NAME 12_raja_launch_transpose
    SOURCES 12_raja_launch_transpose.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()
