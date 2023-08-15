###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (ENABLE_CUDA)
  raja_add_executable(
    NAME fractal-ex2-RAJA-CUDA 
    SOURCES fractal-ex2-RAJA-CUDA.cpp
    DEPENDS_ON cuda writeBMP umpire RAJA)
endif()
