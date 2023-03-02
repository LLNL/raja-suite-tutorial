###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME fractal-ex2-CUDA 
  SOURCES fractal-ex2-CUDA.cpp
  DEPENDS_ON cuda)

raja_add_executable(
  NAME fractal-ex3-RAJA-CUDA 
  SOURCES fractal-ex3-RAJA-CUDA.cpp
  DEPENDS_ON cuda)

add_subdirectory(solutions)
