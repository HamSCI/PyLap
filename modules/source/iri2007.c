#include "../include/pharlap.h"

extern void iri_sub_(int *jf, int *jmag, float *glat, float *glon, int *year,
  int *mmdd, float *dhour, float *heibeg, float *heiend, float *heistep,
  float *outf, float *oarr);

static PyObject *iri2007(PyObject *self, PyObject *args)
{
  Py_ssize_t num_args = PyTuple_Size(args);
  ASSERT((num_args == 4 || num_args == 6), PyExc_ValueError,
    "invalid number of arguments");
  /* Initialize NumPy. */
  import_array();

  /* Input parameters. */
  float glat, glon, r12_idx, heibeg, heiend, heistp;
  int num_heights;
  PyObject *tm, *iri_options;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "fffO!|ffiO!", &glat, &glon, &r12_idx,
      &PyList_Type, &tm, &heibeg, &heistp));

  ASSERT_NOMSG(check_ref_data("iri2007"));

  /* Parse UT. */
  int *ut = (int *)malloc(5 * sizeof(int));
  ASSERT_NOMSG(parse_ut(tm, ut));

  int year = ut[0];
  int mmdd = ut[1] * 100 + ut[2];
  float dhour = (float)(ut[3] + ut[4] / 60.0 + 25.0);

  /* Parse height parameters. */
  if (num_args >= 6) {
    if (num_heights < 2) num_heights = 2;
    if (num_heights > 1000) num_heights = 1000;
    heiend = heibeg + (num_heights - 1) * heistp;
  } else {
    heibeg = 1.f;
    heistp = 1.f;
    heiend = 2.f;
  }

  /* Initialize 'jf' object. */
  int jf[30];

  /* 1. IRI standard settings. */
  for (int i = 0; i < 30; i++) jf[i] = 1;

  jf[4] = jf[5] = 0;
  jf[20] = jf[21] = jf[22] = 0;
  jf[27] = jf[28] = jf[29] = 0;
  /* 2. change some of the standard settings */
  jf[11] = 0;                  /* no messages from IRI */
  jf[20] = 1;                  /* ion drift computed */
  jf[21] = 0;                  /* ion densities in m-3 */
  if (r12_idx > 0 && r12_idx <= 200) {
    jf[25] = 0;                  /* storm model off */
    jf[16] = 0;                  /* user input R12 */
    jf[24] = 0;                  /* user input F10.7 */
    jf[26] = 0;                  /* user input IG12 */
  }
  else if (r12_idx == -1) {
    jf[25] = 1;                  /* storm model on */
    jf[16] = 1;                  /* historical or projected R12 */
    jf[24] = 1;                  /* historical or projected F10.7 */
    jf[26] = 1;                  /* historical of projected IG12 */
  }
  else if (r12_idx == -2) {
    jf[25] = 0;                  /* storm model off */
    jf[16] = 1;                  /* historical or projected R12 */
    jf[24] = 1;                  /* historical or projected F10.7 */
    jf[26] = 1;                  /* historical of projected IG12 */
  }
  else {
    PySys_WriteStderr("Invalid value for R12");
    return;
  }

  /* Initialize iri_sub input and output. */
  float *outf = (float *)malloc(20 * 100 * sizeof(float));
  float *oarr = (float *)malloc(50 * sizeof(float));

  /* Calculate IG12, F10.7 and F10.7_81 from the user input R12 and put into the  
     oarr array for input into iri_sub_. The F10.7 expression is from Davies, 1990, 
     pp 442. The expession for IG12 was obtained from the irisub.for fortran 
     code (line 851) from IRI2012 and was verified against the data found in 
     ig_rz.dat. */
  if (r12_idx > 0) {
    float F107 = 63.75 + r12_idx * (0.728 + r12_idx*0.00089);
    float IG12 = -12.349154 + r12_idx * (1.4683266 - r12_idx * 2.67690893e-03);
    *(oarr + 32) = r12_idx;
    *(oarr + 38) = IG12;
    *(oarr + 40) = F107;
  }

    /* Call the computational subroutine. */
  int jmag = 0;                /* geographic coordinates */
  iri_sub_(jf, &jmag, &glat, &glon, &year, &mmdd, &dhour, &heibeg, &heiend,
           &heistp, outf, oarr);

  /*
   * Create output objects.
   */
  npy_intp outf_dims[] = { 11};
  npy_intp oarr_dims[] = { 100 };

  PyArrayObject *py_outf = (PyArrayObject *)PyArray_SimpleNew(2, outf_dims,
      NPY_DOUBLE);
  PyArrayObject *py_oarr = (PyArrayObject *)PyArray_SimpleNew(1, oarr_dims,
      NPY_DOUBLE);

  for (int i = 0; i < 14; i++) {
    for (int j = 0; j < num_heights; j++) {
      double *f = (double *)PyArray_GETPTR2(py_outf, i, j);

      *f = (double)outf[(j * 20) + i];
    }
  }

  for (int i = 0; i < 100; i++) {
    double *f = (double *)PyArray_GETPTR1(py_oarr, i);

    *f = (double)oarr[i];
  }

  free(outf);
  free(oarr);

  return PyTuple_Pack(2, py_outf, py_oarr);      
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
