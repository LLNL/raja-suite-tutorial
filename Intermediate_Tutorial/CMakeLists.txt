###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME fractal-ex1-serial
  SOURCES fractal-ex1-serial.cpp)

#add_subdirectory(RAJA-BACKENDS)

if(RAJA_ENABLE_CUDA)
  add_subdirectory(CUDA)
endif()

