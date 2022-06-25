#include "../include/pharlap.h"

extern float dop_spread_eq_(float *glon, float *glat, float *year, int *month,
  int *day, int *hour, int *minute, float *r12);

PyObject *dop_spread_eq(PyObject *self, PyObject *args)
{
  /*
   * Check the number of input parameters.
   */
  ASSERT((PyTuple_Size(args) == 4), PyExc_ValueError,
    "invalid number of arguments");

  /*
   * Parse input parameters.
   */
  float lat, lon, r12;
  PyObject *tm;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "ffO!f", &lat, &lon, &PyList_Type, &tm,
      &r12));

  /*
   * Parse UT.
   */
  int *ut = (int *)malloc(5 * sizeof(int));
  ASSERT_NOMSG(parse_ut(tm, ut));

  float year = (float)ut[0];

  /*
   * Call subroutine.
   */
  float result = dop_spread_eq_(&lon, &lat, &year, &(ut[1]), &(ut[2]), &(ut[3]),
      &(ut[4]), &r12);

  return PyFloat_FromDouble((double)result);
}

static PyMethodDef methods[] = {
  { "dop_spread_eq", dop_spread_eq, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "dop_spread_eq",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_dop_spread_eq()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  return m;
}
