#define NO_IMPORT_ARRAY
#include "../../include/pharlap.h"

int parse_tol(PyObject *obj, double *tol, double *step_min, double *step_max)
{
  Py_ssize_t tol_length = PyList_Size(obj);

  if (tol_length == 1) {
    PyObject *tol_id = PyList_GetItem(obj, 0);

    if (PyFloat_Check(tol_id)) {
      double tol_value = PyFloat_AsDouble(tol_id);
      if (tol_value > 1e-12 && tol_value < 1e-2) {
        *tol = tol_value;
        *step_min = 0.01;
        *step_max = 10.0;
      } else {
        PyErr_SetString(PyExc_ValueError, "invalid 'tol' value");
        return 0;
      }
    } else if (PyLong_Check(tol_id)) {
      long tol_value = PyLong_AsLong(tol_id);

      if (tol_value == 1) {
        *tol = 1e-8;
        *step_min = 0.01;
        *step_max = 10.0;
      } else if (tol_value == 2) {
        *tol = 1e-7;
        *step_min = 0.025;
        *step_max = 25.0;
      } else if (tol_value == 3) {
        *tol = 1e-6;
        *step_min = 0.1;
        *step_max = 100.0;
      } else {
        PyErr_SetString(PyExc_ValueError, "invalid 'tol' value");
        return 0;
      }
    } else {
      PyErr_SetString(PyExc_ValueError, "invalid 'tol' value");
      return 0;
    }
  } else if (tol_length == 3) {
    /* FIXME: We don't do any data type checking here... */
    *tol = PyFloat_AsDouble(PyList_GetItem(obj, 0));
    *step_min = PyFloat_AsDouble(PyList_GetItem(obj, 1));
    *step_max = PyFloat_AsDouble(PyList_GetItem(obj, 2));
  } else {
    PyErr_SetString(PyExc_ValueError, "invalid 'tol' value");
    return 0;
  }

  ASSERT_INT((*tol < 1e-12 || *tol > 1e-2 || *step_min > *step_max ||
      *step_min < 0.001 || *step_min > 1 || *step_max > 100 || *step_min < 1),
    PyExc_ValueError, "invalid 'tol' value");

  return 1;
}
