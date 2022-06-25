#include "../include/pharlap.h"

PyObject *raytrace_3d_sp(PyObject *self, PyObject *args)
{
  /* TODO */

  return NULL;
}

static PyMethodDef methods[] = {
  { "raytrace_3d_sp", raytrace_3d_sp, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "raytrace_3d_sp",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_raytrace_3d_sp()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
