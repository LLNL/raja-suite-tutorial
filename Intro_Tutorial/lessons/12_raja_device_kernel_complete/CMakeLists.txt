if (ENABLE_CUDA)
  blt_add_executable(
    NAME twelve
    SOURCES twelve.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()
