if (ENABLE_CUDA) 
  if (RAJA_ENABLE_CALIPER)
    blt_add_executable(
      NAME profile_raja
      SOURCES profile_raja.cpp
      DEPENDS_ON RAJA umpire cuda)
  endif()
endif()
