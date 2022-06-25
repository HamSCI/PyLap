#include "../../include/pharlap.h"

int numpy_to_double_1d(PyArrayObject *arr, double *dst)
{
  PyArray_Descr *dtype = PyArray_DescrFromType(NPY_DOUBLE);

  NpyIter *iter = NpyIter_New(arr, NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP,
      NPY_KEEPORDER, NPY_EQUIV_CASTING, dtype);

  ASSERT_INT_NOMSG((iter != NULL));

  NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);

  if (iternext == NULL) {
    NpyIter_Deallocate(iter);
    return 0;
  }

  char **data_pointer = NpyIter_GetDataPtrArray(iter);
  int idx = 0;

  do {
    char *data = *data_pointer;
    dst[idx] = *(double *)data;
    idx += 1;
  } while (iternext(iter));

  NpyIter_Deallocate(iter);
  return 1;
}

