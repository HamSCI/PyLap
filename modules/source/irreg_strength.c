#include "../include/pharlap.h"

float irreg_strength_(float *glon, float *glat, float *year, int *month,
  int *day, int *hour, int *minute, float *kp, float *dip, float *dec);

PyObject *irreg_strength(PyObject *self, PyObject *args)
{
  Py_ssize_t num_args = PyTuple_Size(args);
  ASSERT((num_args == 4), PyExc_ValueError, "invalid number of arguments");

  /* Initialize NumPy. */
  import_array();

  /* Input parameters. */
  float glat, glon, kp, dip, dec;
  PyObject *tm;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "ffO!f", &glat, &glon, &PyList_Type, &tm,
      &kp));

  /*
   * Parse UT.
   */
  int *ut = (int *)malloc(5 * sizeof(int));
  ASSERT_NOMSG(parse_ut(tm, ut));

  float year = (float)ut[0];

  float out = irreg_strength_(&glon, &glat, &year, &ut[1], &ut[2], &ut[3],
      &ut[4], &kp, &dip, &dec);

  return PyFloat_FromDouble((double)out);
}

static PyMethodDef methods[] = {
  { "irreg_strength", irreg_strength, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "irreg_strength",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_irreg_strength()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
