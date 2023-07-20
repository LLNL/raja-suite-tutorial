###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_executable(
  NAME fractal-ex4-RAJA-launch
  SOURCES fractal-ex4-RAJA-launch.cpp
  DEPENDS_ON RAJA umpire writeBMP)
