#include <stdio.h>

#include "../include/pharlap.h"


extern double abso_bg_(double *lat, double *lon, double *elev, double *freq,
  int *UT, double *R12_index, int *O_mode, int *warn_flg);

static PyObject *abso_bg(PyObject *self, PyObject *args)
{
  char message[256];

  Py_ssize_t num_args = PyTuple_Size(args);
  ASSERT((num_args == 7), PyExc_ValueError, "invalid number of arguments");

  ASSERT_NOMSG(check_ref_data("iri2016"));

  /* Initialize NumPy. */
  import_array();

  /* Input parameters. */
  PyArrayObject *arrs[5];
  PyObject *tm;
  double r12_idx;
  int o_mode, warn_flg;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "O!O!O!O!O!di", &PyArray_Type, &arrs[0],
      &PyArray_Type, &arrs[1], &PyArray_Type, &arrs[2], &PyArray_Type, &arrs[3],
      &PyList_Type, &tm, &r12_idx, &o_mode));

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

  /*
   * Parse UT.
   */
  int *ut = (int *)malloc(5 * sizeof(int));
  ASSERT_NOMSG(parse_ut(tm, ut));

  /*
   * Iterate through each element of the arrays and call `abso_bg_`.
   */
  PyArrayObject *lats = (PyArrayObject *)PyArray_Cast(arrs[0], NPY_DOUBLE);
  PyArrayObject *lons = (PyArrayObject *)PyArray_Cast(arrs[1], NPY_DOUBLE);
  PyArrayObject *elevs = (PyArrayObject *)PyArray_Cast(arrs[2], NPY_DOUBLE);
  PyArrayObject *freqs = (PyArrayObject *)PyArray_Cast(arrs[3], NPY_DOUBLE);

  npy_intp result_dims[1] = { count };
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, result_dims,
      NPY_DOUBLE);

  for (npy_intp i = 0; i < count; i++) {
    double lat = *(double *)PyArray_GETPTR1(lats, i);
    double lon = *(double *)PyArray_GETPTR1(lons, i);
    double elev = *(double *)PyArray_GETPTR1(elevs, i);
    double freq = *(double *)PyArray_GETPTR1(freqs, i);

    double *tmp = (double *)PyArray_GETPTR1(result, i);

    *tmp = abso_bg_(&lat, &lon, &elev, &freq, ut, &r12_idx, &o_mode, &warn_flg);

    if (warn_flg) {
      fprintf(stderr, "absorption potentially unreliable as modified dip > 70 "
        "degrees");
    }
  }

  return result;
}

static PyMethodDef methods[] = {
  { "abso_bg", abso_bg, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "abso_bg",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_abso_bg()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
