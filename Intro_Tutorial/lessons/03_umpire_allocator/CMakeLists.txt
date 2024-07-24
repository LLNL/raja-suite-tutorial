blt_add_executable(
  NAME three
  SOURCES three.cpp
  DEPENDS_ON RAJA umpire cuda)
