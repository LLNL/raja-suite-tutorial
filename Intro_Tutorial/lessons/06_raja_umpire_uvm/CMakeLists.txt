if (ENABLE_CUDA)
  blt_add_executable(
    NAME 06_raja_umpire_uvm
    SOURCES 06_raja_umpire_uvm.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()

