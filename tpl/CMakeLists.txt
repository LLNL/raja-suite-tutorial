# set (RAJA_ENABLE_TESTS Off CACHE BOOL "")
# set (RAJA_ENABLE_EXAMPLES Off CACHE BOOL "")
# set (RAJA_ENABLE_REPRODUCERS Off CACHE BOOL "")
# set (RAJA_ENABLE_EXERCISES Off CACHE BOOL "")
# set (RAJA_ENABLE_DOCUMENTATION Off CACHE BOOL "")
# set (RAJA_ENABLE_BENCHMARKS Off CACHE BOOL "")
# 
# set (UMPIRE_ENABLE_TESTS Off CACHE BOOL "")
# set (UMPIRE_ENABLE_EXAMPLES Off CACHE BOOL "")
# set (UMPIRE_ENABLE_REPRODUCERS Off CACHE BOOL "")
# set (UMPIRE_ENABLE_EXERCISES Off CACHE BOOL "")
# set (UMPIRE_ENABLE_DOCUMENTATION Off CACHE BOOL "")
# set (UMPIRE_ENABLE_BENCHMARKS Off CACHE BOOL "")

add_subdirectory(umpire)
add_subdirectory(raja)

blt_add_library(
    NAME writeBMP
    HEADERS writeBMP.hpp
    SOURCES writeBMP.cpp
    OBJECT TRUE)
