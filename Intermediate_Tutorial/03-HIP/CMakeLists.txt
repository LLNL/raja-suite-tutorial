###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (ENABLE_HIP)
  raja_add_executable(
    NAME fractal-ex3-RAJA-HIP
    SOURCES fractal-ex3-RAJA-HIP.cpp
    DEPENDS_ON blt::hip RAJA umpire writeBMP)
endif()
