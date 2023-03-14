#ifndef UMPIRE_WRITEBMP_H_
#define UMPIRE_WRITEBMP_H_

#include <assert.h>
#include <stdio.h>

class writebmp {
public:
  unsigned char hdr[1078];
  void WriteBMP(int x, int y, unsigned char *bmp, const char * name);
};

#endif //UMPIRE_WRITEBMP_H_
