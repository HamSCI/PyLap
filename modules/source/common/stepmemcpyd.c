#define NO_IMPORT_ARRAY
#include "../../include/pharlap.h"

void stepmemcpyd(double *dest, double *src, int step, int num_vals)
{
  int idx, srcidx;
  
  srcidx = 0;
  for (idx = 0; idx < num_vals; idx++) {
    dest[idx] = src[srcidx];
    srcidx += step;
  }

}