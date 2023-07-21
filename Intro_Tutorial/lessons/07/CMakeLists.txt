if (ENABLE_CUDA)
  blt_add_executable(
    NAME seven
    SOURCES seven.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()
