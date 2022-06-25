#include "../include/pharlap.h"

extern void iri_sub_(int *jf, int *jmag, float *glat, float *glon, int *year,
  int *mmdd, float *dhour, float *heibeg, float *heiend, float *heistep,
  float *outf, float *oarr);

static PyObject *iri2007(PyObject *self, PyObject *args)
{
  return NULL;
}

static PyMethodDef methods[] = {
  { "iri2007", iri2007, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "iri2007",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_iri2007()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
