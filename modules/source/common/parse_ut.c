#define NO_IMPORT_ARRAY
#include "../../include/pharlap.h"

int parse_ut(PyObject *src, int *dst)
{
  ASSERT_INT((PyList_Size(src) == 5), PyExc_ValueError, "invalid 'UT' value");

  for (size_t idx = 0; idx < 5; idx++) {
    long val = PyLong_AsLong(PyList_GetItem(src, idx));
    ASSERT_INT_NOMSG(!PyErr_Occurred());
    dst[idx] = (int)val;
  }

  return 1;
}
