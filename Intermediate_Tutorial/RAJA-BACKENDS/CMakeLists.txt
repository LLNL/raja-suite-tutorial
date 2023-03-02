###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if(RAJA_ENABLE_HIP)
raja_add_executable(
  NAME fractal-ex4-RAJA-HIP
  SOURCES fractal-ex4-RAJA-HIP.cpp
  DEPENDS_ON blt::hip blt::hip_runtime)
endif()

if(RAJA_ENABLE_OPENMP)
raja_add_executable(
  NAME fractal-ex5-RAJA-OpenMP 
  SOURCES fractal-ex5-RAJA-OpenMP.cpp
  DEPENDS_ON openmp)
endif()

add_subdirectory(solutions)
