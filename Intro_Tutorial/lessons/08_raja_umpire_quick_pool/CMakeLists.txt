if (ENABLE_CUDA)
  blt_add_executable(
    NAME 08_raja_umpire_quick_pool
    SOURCES 08_raja_umpire_quick_pool.cpp
    DEPENDS_ON RAJA umpire cuda)

  blt_add_executable(
    NAME 08_raja_umpire_quick_pool_solution
    SOURCES 08_raja_umpire_quick_pool_solution.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()
