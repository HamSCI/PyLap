#include "../include/pharlap.h"

extern void iri_sub_(int *jf, int *jmag, float *glat, float *glon, int *year,
  int *mmdd, float *dhour, float *heibeg, float *heiend, float *heistp,
  float *outf, float *oarr);

static PyObject *iri2012(PyObject *self, PyObject *args)
{
  Py_ssize_t num_args = PyTuple_Size(args);
  ASSERT((num_args == 4 || num_args == 7 || num_args == 8 ||
           num_args == 9 || num_args == 10), PyExc_ValueError,
    "invalid number of arguments");
  /* Initialize NumPy. */
  import_array();

  /* Input parameters. */
  float glat, glon, r12_idx, heibeg, heiend, heistp;
  int num_heights;
  PyObject *tm, *iri_options;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "fffO!|ffiO!", &glat, &glon, &r12_idx,
      &PyList_Type, &tm, &heibeg, &heistp, &num_heights,
      &PyDict_Type, &iri_options));

  ASSERT_NOMSG(check_ref_data("iri2012"));


  /* Parse UT. */
  int *ut = (int *)malloc(5 * sizeof(int));
  ASSERT_NOMSG(parse_ut(tm, ut));

  int year = ut[0];
  int mmdd = ut[1] * 100 + ut[2];
  float dhour = (float)(ut[3] + ut[4] / 60.0 + 25.0);


  /* Parse height parameters. */
  if (num_args >= 5) {
    if (num_heights < 2) num_heights = 2;
    if (num_heights > 1000) num_heights = 1000;
    heiend = heibeg + (num_heights - 1) * heistp;
  } else {
    heibeg = 1.f;
    heistp = 1.f;
    heiend = 2.f;
    num_heights = 0;
  }




  float foF2, hmF2, foF1, hmF1, foE, hmE;
  if (num_args == 10) {
    foF2 = (float)PyFloat_AsDouble(9) + 0;
    hmF2 = (float)PyFloat_AsDouble(9) + 1;
    foF1 = (float)PyFloat_AsDouble(9) + 2;
    hmF1 = (float)PyFloat_AsDouble(9) + 3;
    foE = (float)PyFloat_AsDouble(9) + 4;
    hmE = (float)PyFloat_AsDouble(9) + 5;

    if ((foF2 < 0.1 || foF2 > 100.0) && (foF2 != -1.0)) {
      PySys_WriteStderr("ERROR: Invalid value for input foF2");
      return;
    }
    if ((foF1 < 0.1 || foF1 > 100) && (foF1 != -1)) { 
      PySys_WriteStderr("ERROR: Invalid value for input foF1");
      return;
    }
    if (foF1 > foF2 && foF2 != -1) {
      PySys_WriteStderr("ERROR: foF1 larger than foF2");
      return;
    }
    if ((foE < 0.1 || foE > 100) && (foE != -1)) { 
      PySys_WriteStderr("ERROR: Invalid value for input foE");
      return;
    }
    if (foE > foF1 && foF1 != -1) {
      PySys_WriteStderr("ERROR: foE larger than foF1");
      return;
    }
    if (foE > foF2 && foF2 != -1) {
      PySys_WriteStderr("ERROR: foE larger than foF2");
      return;
    }

    if ((hmF2 < 50 || hmF2 > 1000) && (hmF2 != -1.0)) { 
      PySys_WriteStderr("ERROR: Invalid value for input hmF2");
      return;
    }

    if ((hmF1 < 50 || hmF1 > 1000) && (hmF1 != -1)) { 
      PySys_WriteStderr("ERROR: Invalid value for input hmF1");
      return;
    }
    if (hmF1 > hmF2 && hmF2 != -1) {
      PySys_WriteStderr("ERROR: hmF1 larger than hmF2");
      return;
    }

    if ((hmE < 50 || hmE > 1000) && (hmE != -1)) { 
      PySys_WriteStderr("ERROR: Invalid value for input hmE");
      return;
    }
    if (hmE > hmF1 && hmF1 != -1) {
      PySys_WriteStderr("ERROR: hmE larger than hmF1");
      return;
    }
    if (hmE > hmF2 && hmF2 != -1) {
      PySys_WriteStderr("ERROR: hmE larger than hmF2");
      return;
    }
  } else {
    foF2 = -1;   
    hmF2 = -1;
    foF1 = -1;
    hmF1 = -1;
    foE = -1;
    hmE = -1;
  }



  /* Initialize iri_sub input and output. */
  float *outf = (float *)malloc(20 * 1000 * sizeof(float));
  float *oarr = (float *)malloc(100 * sizeof(float));

  /* Initialize 'jf' object. */
  int jf[38];

  /* 1. IRI standard settings. */
  for (int i = 0; i < 38; i++) jf[i] = 1;

  jf[3] = jf[4] = jf[5] = 0;
  jf[20] = jf[21] = jf[22] = 0;
  jf[27] = jf[28] = jf[29] = 0;
  jf[32] = jf[34] = 0;

  /* 2. A couple customizations. */
  float B0B1_model = (int)PyDict_GetItemString(iri_options, "Ne_B0B1_model");
  float D_model = (int)PyDict_GetItemString(iri_options, "D_model");
  if (B0B1_model == 1) {
    jf[3]  = 0;                  /* B0/B1 - APT-2009 option */
    jf[30] = 1;
  }
  else if (B0B1_model == 2) {
    jf[3]  = 1;                  
    jf[30] = 1;                  /* B0/B1 - Bil-2000 option */
  }
  else {            
    jf[3]  = 0;                  
    jf[30] = 0;                  /* B0/B1 - Gul-1987 option */
  }

  if (D_model == 1) {
    jf[23] = 1;        /*  D-region: IRI-1990 */
  }
  else {
    jf[23] = 0;        /*  D-region: FT-2001 and DRS-1995 */
  }

  jf[33] = 0;                  /* no messages from IRI */
  jf[20] = 1;                  /* ion drift computed */
  jf[21] = 0;                  /* ion densities in m-3 */


  float storm_flag = (int)PyDict_GetItemString(iri_options, "foF2_storm");
  
  if (r12_idx > 0 && r12_idx <= 200) {
    if (storm_flag == 0) {
      jf[25] = 0;                  /* foF2 storm model off */
    }
    else {
      jf[25] = 1;                  /* foF2 storm model on */
    }      
    jf[16] = 0;                  /* user input R12 */
    jf[24] = 0;                  /* user input F10.7 */
    jf[26] = 0;                  /* user input IG12 */
    jf[31] = 0;                  /* user input F10.7_81 */
  }
  else if (r12_idx == -1) {
    if (storm_flag == 0) {
      jf[25] = 0;                  /* foF2 storm model off */
    }
    else {
      jf[25] = 1;                  /* foF2 storm model on */
    }      
    jf[16] = 1;                  /* historical or projected R12 */
    jf[24] = 1;                  /* historical or projected F10.7 */
    jf[26] = 1;                  /* historical or projected IG12 */
    jf[31] = 1;                  /* historical or projected F10.7_81 */
  }
  else {
    PySys_WriteStderr("Invalid value for R12");
    return;
  }
  if (foF2 != -1) {
    jf[7] = 0;
    *(oarr + 0) = foF2;
  }
  if (hmF2 != -1) {
    jf[8] = 0;
    *(oarr + 1) = hmF2;
  }
  if (foF1 != -1) {
    jf[12] = 0;
    *(oarr + 2) = foF1;
  }
  if (hmF1 != -1) {
    jf[13] = 0;
    *(oarr + 3) = hmF1;
  }
  if (foE != -1) {
    jf[14] = 0;
    *(oarr + 4) = foE;
  }
  if (hmE != -1) {
    jf[15] = 0;
    *(oarr + 5) = hmE;
  }

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
    *(oarr + 45) = F107;     /* F10.7_81 is set to F10.7 */
  }

  /* Call the computational subroutine. */
  int jmag = 0;                /* geographic coordinates */
  iri_sub_(jf, &jmag, &glat, &glon, &year, &mmdd, &dhour, &heibeg, &heiend,
           &heistp, outf, oarr);

  /*
   * Create output objects.
   */
  npy_intp outf_dims[] = { 14, num_heights };
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
  { "iri2012", iri2012, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "iri2012",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_iri2012()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
