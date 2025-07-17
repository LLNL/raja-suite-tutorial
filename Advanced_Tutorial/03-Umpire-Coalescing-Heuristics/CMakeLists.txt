###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_executable(
  NAME 03-Umpire-Coalescing-Heuristics
  SOURCES 03-Umpire-Coalescing-Heuristics.cpp
  DEPENDS_ON cuda umpire)
