###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

add_subdirectory(00-BASE)

add_subdirectory(01-SEQ)

if(RAJA_ENABLE_CUDA)
  add_subdirectory(02-CUDA)
  add_subdirectory(04-LAUNCH)
endif()

if(RAJA_ENABLE_HIP)
  add_subdirectory(03-HIP)
  add_subdirectory(04-LAUNCH)
endif()


