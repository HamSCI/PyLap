#include "../../include/pharlap.h"

int parse_elev_freq(PyArrayObject *elevs_src, PyArrayObject *freqs_src,
  double *elevs_dst, double *freqs_dst)
{
  PyArrayObject *arrs[2];
  arrs[0] = elevs_src;
  arrs[1] = freqs_src;

  npy_uint32 arr_flags[2];
  arr_flags[0] = arr_flags[1] = NPY_ITER_READONLY;

  PyArray_Descr *dtypes[2];
  dtypes[0] = dtypes[1] = PyArray_DescrFromType(NPY_DOUBLE);

  NpyIter *iter = NpyIter_MultiNew(3, arrs, NPY_ITER_EXTERNAL_LOOP,
      NPY_KEEPORDER, NPY_EQUIV_CASTING, arr_flags, dtypes);

  ASSERT_INT_NOMSG((iter != NULL));

  NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);

  if (iternext == NULL) {
    NpyIter_Deallocate(iter);
    return 0;
  }

  char **data_pointer_array = NpyIter_GetDataPtrArray(iter);
  int idx = 0;

  do {
    elevs_dst[idx] = *(double *)data_pointer_array[0];
    freqs_dst[idx] = *(double *)data_pointer_array[1];

    idx += 1;
  } while (iternext(iter));

  NpyIter_Deallocate(iter);
  return 1;
}
