#include "../include/pharlap.h"


extern void igrf_calc_(float *glat, float *glon, float *dec_year, float *height,
  float *dipole_moment, float *babs, float *bnorth, float *beast, float *bdown,
  float *dip, float *dec, float *dip_lat, float *l_value, int *l_value_code);

extern int julday_(int *day, int *month, int *year);

static PyObject *igrf2007(PyObject *self, PyObject *args)
{
  return NULL;
}

static PyMethodDef methods[] = {
  { "igrf2007", igrf2007, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "igrf2007",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_igrf2007()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
