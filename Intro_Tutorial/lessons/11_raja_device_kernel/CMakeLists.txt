if (ENABLE_CUDA)
  blt_add_executable(
    NAME 11_raja_device_kernel
    SOURCES 11_raja_device_kernel.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()
