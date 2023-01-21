cmake_minimum_required(VERSION 3.19)

project(RajaSuiteTutorial LANGUAGES C CXX)

set (ENABLE_TESTS Off CACHE BOOL "")
set (ENABLE_EXAMPLES Off CACHE BOOL "")
set (ENABLE_REPRODUCERS Off CACHE BOOL "")
set (ENABLE_EXERCISES Off CACHE BOOL "")
set (ENABLE_DOCUMENTATION Off CACHE BOOL "")
set (ENABLE_BENCHMARKS Off CACHE BOOL "")

set(BLT_CXX_STD "c++14")
include(blt/SetupBLT.cmake)

add_subdirectory(tpl)
add_subdirectory(lessons)
