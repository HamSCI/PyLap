#include "../include/pharlap.h"

extern void iri2016_calc_(int *jf, int *jmag, float *glat, float *glon,
  int *year, int *mmdd, float *dhour, float *heibeg, float *heiend,
  float *heistep, float *outf, float *oarr);

static PyObject *iri2016(PyObject *self, PyObject *args)
{
  Py_ssize_t num_args = PyTuple_Size(args);
  ASSERT((num_args == 4 || num_args == 7 || num_args == 8), PyExc_ValueError,
    "invalid number of arguments");
  /* Initialize NumPy. */
  import_array();

  /* Input parameters. */
  float glat, glon, r12_idx, height_start, height_end, height_step;
  int num_heights;
  PyObject *tm, *iri_options;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "fffO!|ffiO!", &glat, &glon, &r12_idx,
      &PyList_Type, &tm, &height_start, &height_step, &num_heights,
      &PyDict_Type, &iri_options));

  ASSERT_NOMSG(check_ref_data("iri2016"));

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
    height_end = height_start + (num_heights - 1) * height_step;
  } else {
    height_start = 1.f;
    height_step = 1.f;
    height_end = 2.f;
    num_heights = 0;
  }

  /* Initialize iri_sub input and output. */
  float *outf = (float *)malloc(20 * 1000 * sizeof(float));
  float *oarr = (float *)malloc(100 * sizeof(float));

  /* Initialize 'jf' object. */
  int jf[50];

  /* 1. IRI standard settings. */
  for (int i = 0; i < 50; i++) jf[i] = 1;

  jf[3] = jf[4] = jf[5] = 0;
  jf[20] = jf[21] = jf[22] = 0;
  jf[27] = jf[28] = jf[29] = 0;
  jf[32] = jf[34] = jf[38] = 0;
  jf[46] = jf[47] = jf[48] = jf[49] = 0;

  /* 2. A couple customizations. */
  jf[33] = jf[21] = 0;
  jf[20] = jf[27] = 1;

  if (r12_idx > 0.0 && r12_idx <= 200.0) {
    jf[16] = jf[24] = jf[25] = jf[26] = jf[31] = 0;

    float f107 = 63.75 + r12_idx * (0.728 + r12_idx * 0.00089);
    float ig12 = -12.349154 + r12_idx * (1.4683266 - r12_idx * 2.67690893e-03);
    oarr[32] = r12_idx;
    oarr[38] = ig12;
    oarr[40] = f107;
    oarr[45] = f107;
  } else if (r12_idx == -1.0) {
    jf[16] = jf[24] = jf[25] = jf[26] = jf[31] = 1;
  } else {
    PyErr_SetString(PyExc_ValueError, "invalid value for R12");
    return NULL;
  }

  if (num_args == 8) {
    PyObject *obj;
    const char *str;

    float foF2, hmF2, foF1, hmF1, foE, hmE;
    float b0, b1;
    float hnea, hnee;

    int foF2_valid = 0, foF1_valid = 0, foE_valid = 0, hmF2_valid = 0,
        hmF1_valid = 0, hmE_valid = 0;
    int num_valid_fields = 0;
    int invalid_field_flag = 0;

    /* Read 'iri_messages' field. */
    obj = PyDict_GetItemString(iri_options, "iri_messages");
  
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "off") == 0) { jf[33] = 0; }
      else if (strcmp(str, "on") == 0) { jf[33] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'foF2' field. */
    obj = PyDict_GetItemString(iri_options, "foF2");
    if(obj!=NULL){
      if (isdigit(obj) == 1 && !PyFloat_Check(obj)) {
        num_valid_fields += 1;
        foF2 = (float)PyFloat_AsDouble(obj);
        jf[7] = 0;
       *(oarr + 0) = foF2;
      } else {
       invalid_field_flag = 1;
    }
  }
    /* Read 'hmF2' field. */
    obj = PyDict_GetItemString(iri_options, "hmF2");
    if(obj != NULL){
      if (isdigit(obj) == 1 && !PyFloat_Check(obj)) {
        num_valid_fields += 1;
       hmF2 = (float)PyFloat_AsDouble(obj);
        jf[8] = 0;
        *(oarr + 1) = hmF2;
      } else {
        invalid_field_flag = 1;
    }
    }

    /* Read 'foF1' field. */
    obj = PyDict_GetItemString(iri_options, "foF1");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      foF1 = (float)PyFloat_AsDouble(obj);
      jf[12] = 0;
      *(oarr + 2) = foF1;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'hmF1' field. */
    obj = PyDict_GetItemString(iri_options, "hmF1");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      hmF1 = (float)PyFloat_AsDouble(obj);
      jf[13] = 0;
      *(oarr + 3) = hmF1;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'foE' field. */
    obj = PyDict_GetItemString(iri_options, "foE");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      foE = (float)PyFloat_AsDouble(obj);
      jf[14] = 0;
      *(oarr + 4) = foE;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'hmE' field. */
    obj = PyDict_GetItemString(iri_options, "hmE");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      hmE = (float)PyFloat_AsDouble(obj);
      jf[15] = 0;
      *(oarr + 5) = hmE;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'B0' field. */
    obj = PyDict_GetItemString(iri_options, "B0");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      b0 = (float)PyFloat_AsDouble(obj);
      jf[42] = 0;
      *(oarr + 9) = b0;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'B1' field. */
    obj = PyDict_GetItemString(iri_options, "B1");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      b1 = (float)PyFloat_AsDouble(obj);
      jf[43] = 0;
      *(oarr + 34) = b1;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'HNEA' field (Ne lower boundary). */
    obj = PyDict_GetItemString(iri_options, "HNEA");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      hnea = (float)PyFloat_AsDouble(obj);
      jf[44] = 0;
      *(oarr + 88) = hnea;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'HNEE' field (Ne upper boundary). */
    obj = PyDict_GetItemString(iri_options, "HNEE");
    if (obj != NULL && PyFloat_Check(obj)) {
      num_valid_fields += 1;
      hnee = (float)PyFloat_AsDouble(obj);
      jf[45] = 0;
      *(oarr + 89) = hnee;
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'foF2_coeffs' field. */
    obj = PyDict_GetItemString(iri_options, "foF2_coeffs");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "URSI") == 0) { jf[4] = 0; }
      else if (strcmp(str, "CCIR") == 0) { jf[4] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Ni_model' field. */
    obj = PyDict_GetItemString(iri_options, "Ni_model");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "RBV-2010 & TTS-2005") == 0) {
        jf[5] = 0;
      } else if (strcmp(str, "DS-1995 & DY-1985") == 0) {
        jf[5] = 1;
      } else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Te_profile' field. */
    obj = PyDict_GetItemString(iri_options, "Te_profile");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "Te/Ne correlation") == 0) { jf[9] = 0; }
      else if (strcmp(str, "standard") == 0) { jf[9] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Te_topside_model' field. */
    obj = PyDict_GetItemString(iri_options, "Te_topside_model");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "TBT-2012") == 0) { jf[22] = 0; }
      else if (strcmp(str, "Bil-1985") == 0) { jf[22] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Te_PF107_dependence' field. */
    obj = PyDict_GetItemString(iri_options, "Te_PF107_dependence");

    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "off") == 0) { jf[41] = 0; }
      else if (strcmp(str, "on") == 0) { jf[41] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Ne_tops_limited' field. */
    obj = PyDict_GetItemString(iri_options, "Ne_tops_limited");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "f10.7 unlimited") == 0) { jf[6] = 0; }
      else if (strcmp(str, "f10.7 limited") == 0) { jf[6] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Ne_profile_calc' field. */
    obj = PyDict_GetItemString(iri_options, "Ne_profile_calc");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "Lay-function") == 0) { jf[10] = 0; }
      else if (strcmp(str, "standard") == 0) { jf[10] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Ne_B0B1_model' field. */
    obj = PyDict_GetItemString(iri_options, "Ne_B0B1_model");

    if (obj != NULL && PyUnicode_Check(obj)) {
    //printf("Within the NE_B0B1_model");

      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "ABT-2009") == 0) { jf[3] = 0; jf[30] = 1; }
      else if (strcmp(str, "Bil-2000") == 0) { jf[3] = 1; jf[30] = 1; }
      else if (strcmp(str, "Gul-1987") == 0) { jf[3] = 0; jf[30] = 0; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'Ne_topside_model' field. */
    obj = PyDict_GetItemString(iri_options, "Ne_topside_model");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "IRI-2001") == 0) { jf[28] = 1; jf[29] = 1; }
      else if (strcmp(str, "IRI-2001 corrected") == 0) { jf[28] = 0; jf[29] = 1; }
      else if (strcmp(str, "NeQuick") == 0) { jf[28] = 0; jf[29] = 0; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'F1_model' field. */
    obj = PyDict_GetItemString(iri_options, "F1_model");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "Scotto-1997 no L") == 0) { jf[18] = 1; jf[19] = 1; }
      else if (strcmp(str, "Scotto-1997 with L") == 0) { jf[18] = 1; jf[19] = 0; }
      else if (strcmp(str, "solar zenith") == 0) { jf[18] = 0; jf[19] = 1; }
      else if (strcmp(str, "none") == 0) { jf[18] = 0; jf[19] = 0; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'D_model' field. */
    obj = PyDict_GetItemString(iri_options, "D_model");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "FT-2001") == 0) { jf[23] = 0; }
      else if (strcmp(str, "IRI-1990") == 0) { jf[23] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'hmF2_model' field. */
    obj = PyDict_GetItemString(iri_options, "hmF2_model");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "AMTB") == 0) { jf[38] = 0; jf[39] = 1; }
      else if (strcmp(str, "Shubin-COSMIC") == 0) { jf[38] = 0; jf[39] = 0; }
      else if (strcmp(str, "M3000F2") == 0) { jf[38] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'foF2_storm' field. */
    obj = PyDict_GetItemString(iri_options, "foF2_storm");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "off") == 0) { jf[25] = 0; }
      else if (strcmp(str, "on") == 0) { jf[25] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'hmF2_storm' field. */
    obj = PyDict_GetItemString(iri_options, "hmF2_storm");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "off") == 0) { jf[35] = 0; }
      else if (strcmp(str, "on") == 0) { jf[35] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'foE_storm' field. */
    obj = PyDict_GetItemString(iri_options, "foE_storm");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "off") == 0) { jf[34] = 0; }
      else if (strcmp(str, "on") == 0) { jf[34] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'topside_storm' field. */
    obj = PyDict_GetItemString(iri_options, "topside_storm");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "off") == 0) { jf[36] = 0; }
      else if (strcmp(str, "on") == 0) { jf[36] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'auroral_boundary_model' field. */
    obj = PyDict_GetItemString(iri_options, "auroral_boundary_model");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "off") == 0) { jf[32] = 0; }
      else if (strcmp(str, "on") == 0) { jf[32] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }

    /* Read 'covington' field. */
    obj = PyDict_GetItemString(iri_options, "covington");
    if (obj != NULL && PyUnicode_Check(obj)) {
      num_valid_fields += 1;
      str = PyUnicode_AsUTF8(obj);

      if (strcmp(str, "IG12") == 0) { jf[40] = 0; }
      else if (strcmp(str, "F10.7_12") == 0) { jf[40] = 1; }
      else { invalid_field_flag = 1; }
    } else {
      invalid_field_flag = 1;
    }
//printf(()num_valid_fields);
//printf((str)PyDict_Size(iri_options));
// PySys_WriteStderr("",num_valid_fields);
// PySys_WriteStderr("",(int)PyDict_Size(iri_options));

    if (num_valid_fields != (int)PyDict_Size(iri_options)) {
      PySys_WriteStderr("Warning: IRI2016 - Some of the fields of the supplied "
        "iri_options structure \n");
      PySys_WriteStderr(" are not valid fields. These fields have been "
        "ignored.\n\n");
    }

    // if (invalid_field_flag) {
    //   PySys_WriteStderr("Warning: IRI2016 - Some of the fields of the supplied "
    //     "iri_options structure\n");
    //   PySys_WriteStderr("         have invalid values. These fields have been  "
    //     "ignored.\n\n");
    // }

    ASSERT(!(foF2_valid && (foF2 < 0.1f || foF2 > 100.f)), PyExc_ValueError,
      "ERROR: Invalid value for input foF2");

    ASSERT(!(foF1_valid && (foF1 < 0.1f || foF1 > 100.f)), PyExc_ValueError,
      "ERROR: Invalid value for input foF1");

    ASSERT(!(foF2_valid && foF1_valid && foF1 > foF2), PyExc_ValueError,
      "ERROR: foF1 larger than foF2");

    ASSERT(!(foE_valid && (foE < 0.1f || foE > 100.f)), PyExc_ValueError,
      "ERROR: Invalid value for foE");

    ASSERT(!(foF1_valid && foE_valid && foE > foF1), PyExc_ValueError,
      "ERROR: foE larger than foF1");

    ASSERT(!(foF2_valid && foE_valid && foE > foF2), PyExc_ValueError,
      "ERROR: foE larger than foF2");

    ASSERT(!(hmF2_valid && (hmF2 < 50.f || hmF2 > 1000.f)), PyExc_ValueError,
      "ERROR: Invalid value for input hmF2");

    ASSERT(!(hmF1_valid && (hmF1 < 50.f || hmF1 > 1000.f)), PyExc_ValueError,
      "ERROR: Invalid value for input hmF1");

    ASSERT(!(hmF2_valid && hmF1_valid && hmF1 > hmF2), PyExc_ValueError,
      "ERROR: hmF1 larger than hmF2");

    ASSERT(!(hmE_valid && (hmE < 50.f || hmE > 1000.f)), PyExc_ValueError,
      "ERROR: Invalid value for input hmE");

    ASSERT(!(hmF1_valid && hmE_valid && hmE > hmF1), PyExc_ValueError,
      "ERROR: hmE larger than hmF1");

    ASSERT(!(hmF2_valid && hmE_valid && hmE > hmF2), PyExc_ValueError,
      "ERROR: hmE larger than hmF2");
  }

  int jmag = 0;

  iri2016_calc_(jf, &jmag, &glat, &glon, &year, &mmdd, &dhour, &height_start,
    &height_end, &height_step, outf, oarr);

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
  { "iri2016", iri2016, METH_VARARGS, "" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "iri2016",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_iri2016()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred()) return NULL;
  import_array();
  return m;
}
