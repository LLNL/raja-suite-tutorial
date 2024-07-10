###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_executable(
  NAME 02-CAMP-resource
  SOURCES 02-CAMP-resource.cpp
  DEPENDS_ON cuda umpire RAJA)
