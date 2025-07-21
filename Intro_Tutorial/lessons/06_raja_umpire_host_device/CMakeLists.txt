if (ENABLE_CUDA)
  blt_add_executable(
    NAME 06_raja_umpire_host_device
    SOURCES 06_raja_umpire_host_device.cpp
    DEPENDS_ON RAJA umpire cuda)

  blt_add_executable(
    NAME 06_raja_umpire_host_device_solution
    SOURCES 06_raja_umpire_host_device_solution.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()
