#ifndef UMPIRE_WRITEBMP_H_
#define UMPIRE_WRITEBMP_H_

#include <assert.h>
#include <stdio.h>

class writebmp {
public:
  const unsigned char bmphdr[54] = {66, 77, 255, 255, 255, 255, 0, 0, 0, 0, 54, 4, 0, 0, 40, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 8, 0, 0, 0, 0, 0, 255, 255, 255, 255, 196, 14, 0, 0, 196, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  unsigned char hdr[1078];
  int i, j, c, xcorr, diff;
  FILE *f;

  // Write a .bmp image file with the computed pixels
  // input: width and height, pointer to the computed pixels and an output file name
  // output: .bmp file under the name specified with the resulting fractal image
  void WriteBMP(int x, int y, unsigned char *bmp, const char * name);
};

#endif //UMPIRE_WRITEBMP_H_
