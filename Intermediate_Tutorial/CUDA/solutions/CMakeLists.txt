###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME fractal-solution-CUDA 
  SOURCES fractal-solution-CUDA.cpp
  DEPENDS_ON cuda)

raja_add_executable(
  NAME fractal-solution-RAJA-CUDA
  SOURCES fractal-solution-RAJA-CUDA.cpp
  DEPENDS_ON cuda)
