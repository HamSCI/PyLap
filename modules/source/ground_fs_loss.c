#include "../include/pharlap.h"

double forward_scatter_loss_(double *lat, double *lon, double *elev,
  double *freq);

PyObject *ground_fs_loss(PyObject *self, PyObject *args)
{
  char message[256];

  Py_ssize_t num_args = PyTuple_Size(args);
  ASSERT((num_args == 4), PyExc_ValueError, "invalid number of arguments");

  /* Initialize NumPy. */
  import_array();

  PyArrayObject *arrs[5];

  ASSERT_NOMSG(PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &arrs[0],
      &PyArray_Type, &arrs[1], &PyArray_Type, &arrs[2], &PyArray_Type,
      &arrs[3]));

  /* We need to explicitly define this to be NULL, otherwise Python throws a
   * non-deterministic fit. */
  arrs[4] = NULL;

  /*
   * Validate numpy array parameters.
   */
  npy_intp count;

  for (int i = 0; i < 4; i++) {
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

  PyArrayObject *lats = (PyArrayObject *)PyArray_Cast(arrs[0], NPY_DOUBLE);
  PyArrayObject *lons = (PyArrayObject *)PyArray_Cast(arrs[1], NPY_DOUBLE);
  PyArrayObject *elevs = (PyArrayObject *)PyArray_Cast(arrs[2], NPY_DOUBLE);
  PyArrayObject *freqs = (PyArrayObject *)PyArray_Cast(arrs[3], NPY_DOUBLE);

  npy_intp out_dims[1] = { count };
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, out_dims,
      NPY_DOUBLE);

  for (npy_intp i = 0; i < count; i++) {
    double lat = *(double *)PyArray_GETPTR1(lats, i);
    double lon = *(double *)PyArray_GETPTR1(lons, i);
    double elev = *(double *)PyArray_GETPTR1(elevs, i);
    double freq = *(double *)PyArray_GETPTR1(freqs, i);

    double *tmp = (double *)PyArray_GETPTR1(result, i);

    *tmp = forward_scatter_loss_(&lat, &lon, &elev, &freq);
  }

  return (PyObject *)result;
}

static PyMethodDef methods[] = {
  { "ground_fs_loss", ground_fs_loss, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "ground_fs_loss",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_ground_fs_loss()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
