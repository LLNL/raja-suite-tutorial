###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_executable(
  NAME fractal-ex1-RAJA-seq
  SOURCES fractal-ex1-RAJA-seq.cpp
  DEPENDS_ON RAJA umpire writeBMP)
