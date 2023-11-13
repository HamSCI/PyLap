#ifndef PHARLAP_H_
#define PHARLAP_H_

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #include <numpy/arrayobject.h>

/* if `op` expression evaluates to a falsy value, raise an exception in Python
 * and return NULL. */
#define ASSERT(op, typ, msg) \
  if (!op) { PyErr_SetString(typ, msg); return NULL; }

#define ASSERT_NOMSG(op) if (!op) return NULL;

#define ASSERT_INT(op, typ, msg) \
  if (!op) { PyErr_SetString(typ, msg); return 0; }

#define ASSERT_INT_NOMSG(op) if (!op) return 0;

/*
 * Utility functions
 */
int check_ref_data(const char *);
int parse_elev_freq(PyArrayObject *, PyArrayObject *, double *, double *);
int parse_tol(PyObject *, double *, double *, double *);
int parse_ut(PyObject *, int *);

void stepmemcpyd(double *dst, double *src, int stride, int count);

#endif /* PHARLAP_H_ */
