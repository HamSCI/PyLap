#include "../include/pharlap.h"

extern void land_sea_(double *lat, double *lon, char *land_type);

PyObject *ground_bs_loss(PyObject *self, PyObject *args)
{
  char message[256];

  Py_ssize_t num_args = PyTuple_Size(args);
  ASSERT((num_args == 2), PyExc_ValueError, "invalid number of arguments");

  /* Initialize NumPy. */
  import_array();

  /* Input parameters. */
  PyArrayObject *arrs[3];

  ASSERT_NOMSG(PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arrs[0],
      &PyArray_Type, &arrs[1]));

  /* We need to explicitly define this to be NULL, otherwise Python throws a
   * non-deterministic fit. */
  arrs[2] = NULL;

  /*
   * Validate numpy array parameters.
   */
  npy_intp count;

  for (int i = 0; i < 2; i++) {
    if (PyArray_NDIM(arrs[i]) != 1) {
      snprintf(message, 256, "parameter %d is invalid", i);
      PyErr_SetString(PyExc_ValueError, message);
      return NULL;
    }

    if (i == 0) {
      count = PyArray_DIM(arrs[i], 0);
      continue;
    }

    if (PyArray_DIM(arrs[i], 0) != count) {
      snprintf(message, 256, "parameter %d is not the correct size", i);
      PyErr_SetString(PyExc_ValueError, message);
      return NULL;
    }
  }

  ASSERT_NOMSG(check_ref_data("land_sea"));

  npy_uint32 arr_flags[3];
  arr_flags[0] = arr_flags[1] = NPY_ITER_READONLY;
  arr_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

  PyArray_Descr *dtype[3];
  dtype[0] = dtype[1] = dtype[2] = PyArray_DescrFromType(NPY_DOUBLE);

  NpyIter *iter = NpyIter_MultiNew(3, arrs, 0, NPY_FORTRANORDER,
      NPY_EQUIV_CASTING, arr_flags, dtype);

  ASSERT_NOMSG((iter != NULL));

  NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);

  if (iternext == NULL) {
    NpyIter_Deallocate(iter);
    return NULL;
  }

  char **data_pointer_array = NpyIter_GetDataPtrArray(iter);

  do {
    double lat = *(double *)data_pointer_array[0];
    double lon = *(double *)data_pointer_array[1];
    double *out = (double *)data_pointer_array[2];

    /* Start by assuming we're on the sea. */
    char land_type[2];
    land_type[0] = 'S';
    land_type[1] = '\0';
    double back_scatter_loss = 0.0025;

    land_sea_(&lat, &lon, land_type);

    /* If land_sea_() indicates we're on land, half the backscatter loss. */
    if (strcmp(land_type, "L") == 0) {
      back_scatter_loss = back_scatter_loss * 0.5;
    }

    *out = -10.0 * log10(back_scatter_loss);
  } while (iternext(iter));

  PyObject *result = (PyObject *)NpyIter_GetOperandArray(iter)[2];
  Py_INCREF(result);

  if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
    Py_DECREF(result);
    return NULL;
  }

  return result;
}

static PyMethodDef methods[] = {
  { "ground_bs_loss", ground_bs_loss, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "ground_bs_loss",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_ground_bs_loss()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
