###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if(RAJA_ENABLE_HIP)
raja_add_executable(
  NAME fractal-RAJA-HIP
  SOURCES fractal-RAJA-HIP.cpp
  DEPENDS_ON blt::hip blt::hip_runtime)
endif()

