#include <stdio.h>

#include "../include/pharlap.h"

extern void nrlmsise00_calc_(int *iyd, float *sec, float *alt, float *glat,
  float *glon, float *stl, float *f107_81, float *f107_prior_day,
  float *ap_daily, int *mass, float *d, float *t);

extern void apfmsis_call_(int *ut, float *f107_day, float *f107_prior_day,
  float *f107_81, float *f107_365, float *ap);

extern int julday_(int *day, int *month, int *year);

static PyObject *nrlmsise00(PyObject *self, PyObject *args)
{
  char message[256];

  Py_ssize_t num_args = PyTuple_Size(args);

  ASSERT((num_args == 4 || num_args == 7), PyExc_ValueError, "invalid number "
    "of arguments");

  ASSERT_NOMSG(check_ref_data("iri2016"));

  /* Initialize NumPy. */
  import_array();

  /* Input parameters. */
  PyArrayObject *arrs[3], *densities, *temps;
  PyObject *tm;
  float f107_prior_day, f107_81, ap_daily;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "O!O!O!O!|fff", &PyArray_Type, &arrs[0],
      &PyArray_Type, &arrs[1], &PyArray_Type, &arrs[2], &PyList_Type, &tm,
      &f107_prior_day, &f107_81, &ap_daily));

  /*
   * Validate array shapes.
   */
  npy_intp count;

  for (int i = 0; i < 3; i++) {
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

  /* We need to explicitly define this to be NULL, otherwise Python throws a
   * non-deterministic fit. */
  npy_intp densities_dims[2] = { 9L, (long)count };
  npy_intp temps_dims[2] = { 2L, (long)count };

  densities = (PyArrayObject *)PyArray_SimpleNew(2, densities_dims, NPY_FLOAT);
  temps = (PyArrayObject *)PyArray_SimpleNew(2, temps_dims, NPY_FLOAT);

  /*
   * Parse UT.
   */
  int *ut = (int *)malloc(5 * sizeof(int));
  ASSERT_NOMSG(parse_ut(tm, ut));

  int year = ut[0];
  int month = ut[1];
  int day = ut[2];
  int hour = ut[3];
  int minute = ut[4];

  int dd1 = 0;
  int mm1 = 1;

  int day_of_year = julday_(&day, &month, &year) - julday_(&dd1, &mm1, &year);

  int iyd = (year % 100) * 1000 + day_of_year;
  float sec = minute * 60 + hour * 60 * 60;

  /*
   * Handle optional arguments.
   */
  if (num_args != 7) {
    float f107_day, f107_365, ap[7];

    apfmsis_call_(ut, &f107_day, &f107_prior_day, &f107_81, &f107_365, ap);
    ap_daily = ap[1];
  }

  int mass = 48;

  PyArrayObject *lats_f = (PyArrayObject *)PyArray_Cast(arrs[0], NPY_FLOAT);
  PyArrayObject *lons_f = (PyArrayObject *)PyArray_Cast(arrs[1], NPY_FLOAT);
  PyArrayObject *alts_f = (PyArrayObject *)PyArray_Cast(arrs[2], NPY_FLOAT);

  for (int i = 0; i < count; i++) {
    float lat = *(float *)PyArray_GETPTR1(lats_f, i);
    float lon = *(float *)PyArray_GETPTR1(lons_f, i);
    float alt = *(float *)PyArray_GETPTR1(alts_f, i);

    float stl = (float)hour + ((float)minute) / 60.f + lon / 15.f;

    float *density = (float *)malloc(9 * sizeof(float));
    float *temp = (float *)malloc(2 * sizeof(float));

    if (stl > 24.f) stl -= 24.f;
    if (stl < 0.f) stl += 24.f;

    nrlmsise00_calc_(&iyd, &sec, &alt, &lat, &lon, &stl, &f107_81,
      &f107_prior_day, &ap_daily, &mass, density, temp);

    for (int j = 0; j < 9; j++) {
      float *tmp = (float *)PyArray_GETPTR2(densities, j, i);
      *tmp = density[j];
    }

    for (int j = 0; j < 2; j++) {
      float *tmp = (float *)PyArray_GETPTR2(temps, j, i);
      *tmp = temp[j];
    }

    free(density);
    free(temp);
  }

  return PyTuple_Pack(2, densities, temps);
}

static PyMethodDef methods[] = {
  { "nrlmsise00", nrlmsise00, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "nrlmsise00",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_nrlmsise00()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
