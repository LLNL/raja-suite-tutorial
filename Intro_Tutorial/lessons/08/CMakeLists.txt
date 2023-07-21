if (ENABLE_CUDA)
  blt_add_executable(
    NAME eight
    SOURCES eight.cpp
    DEPENDS_ON RAJA umpire cuda)
elseif (ENABLE_HIP)
  blt_add_executable(
    NAME eight
    SOURCES eight.cpp
    DEPENDS_ON RAJA umpire blt::hip)
endif()
