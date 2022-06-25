#include "../include/pharlap.h"

extern void igrf2016_calc_(float *glat, float *glon, float *dec_year,
  float *height, float *dipole_moment, float *babs, float *bnorth, float *beast,
  float *bdown, float *dip, float *dec, float *dip_lat, float *l_value,
  int *l_value_code);

extern int julday_(int *day, int *month, int *year);

static PyObject *igrf2016(PyObject *self, PyObject *args)
{
  /* Initialize NumPy. */
  import_array();

  Py_ssize_t num_args = PyTuple_Size(args);

  ASSERT((num_args == 4), PyExc_ValueError,
    "4 inputs are required when calling igrf2016");

  float latitude, longitude, height;
  PyObject *tm;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "ffO!f", &latitude, &longitude,
      &PyList_Type, &tm, &height));

  ASSERT_NOMSG(check_ref_data("iri2016"));

  /* Parse UT. */
  int *ut = (int *)malloc(5 * sizeof(int));
  ASSERT_NOMSG(parse_ut(tm, ut));

  /* Compute decimal year. */
  int year = ut[0];
  int month = ut[1];
  int day = ut[2];
  int hour = ut[3];
  int dd1 = 0;
  int dd2 = 21;
  int mm1 = 1;
  int mm2 = 12;

  int day_of_year = julday_(&day, &month, &year) - julday_(&dd1, &mm1, &year);
  int days_in_year = julday_(&dd2, &mm2, &year) - julday_(&dd1, &mm1, &year);

  float dec_year = (float)year + (float)day_of_year / (float)days_in_year +
    (float)hour / (24 * (float)days_in_year);

  float dipole_moment, babs, bnorth, beast, bdown, dip, dec, dip_lat, l_value;
  int l_value_code;

  /* Call subroutine. */
  igrf2016_calc_(&latitude, &longitude, &dec_year, &height, &dipole_moment,
    &babs, &bnorth, &beast, &bdown, &dip, &dec, &dip_lat, &l_value,
    &l_value_code);

  npy_intp out_dims[] = { 10 };

  PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(1, out_dims,
      NPY_DOUBLE);

  *(double *)PyArray_GETPTR1(output, 0) = bnorth / 10000.0;
  *(double *)PyArray_GETPTR1(output, 1) = beast / 10000.0;
  *(double *)PyArray_GETPTR1(output, 2) = bdown / 10000.0;
  *(double *)PyArray_GETPTR1(output, 3) = babs / 10000.0;
  *(double *)PyArray_GETPTR1(output, 4) = dipole_moment;
  *(double *)PyArray_GETPTR1(output, 5) = l_value;
  *(double *)PyArray_GETPTR1(output, 6) = l_value_code;
  *(double *)PyArray_GETPTR1(output, 7) = dip;
  *(double *)PyArray_GETPTR1(output, 8) = dip_lat;
  *(double *)PyArray_GETPTR1(output, 9) = dec;

  return (PyObject *)output;
}

static PyMethodDef methods[] = {
  { "igrf2016", igrf2016, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "igrf2016",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_igrf2016()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
