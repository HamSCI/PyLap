#include <stdlib.h>

#include "../include/pharlap.h"
#include <iono_structures.h>

#define MAX_POINTS_IN_RAY 20000

extern void raytrace_2d_(double *origin_lat, double *origin_lon, int *num_rays, double *elevs,
                         double *bearings, double *freqs, int *nhops, double *step_size_min, double *step_size_max,
                         double *tol, struct ionosphere_struct *ionosphere, int *irregs_flag, int *return_ray_path_data,
                         int *return_ray_state_vec, double *ray_data, double *ray_path_data, int *ray_label,
                         int *nhops_attempted, double *ray_state_vec_in, int *npts_in_ray, double *ray_state_vec_out,
                         double *elapsed_time);

static void verifyIonoGrid(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                           PyArrayObject *irreg_grid);

static PyObject *buildOutput(int num_rays, int *nhops_attempted, int *npts_in_ray, double *ray_data,
                             double *ray_path_data, double *freqs_data, double *elevs_data, int *ray_label, double *ray_state_vec_out);

static void buidlIonoStruct(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                            PyArrayObject *irreg_grid, double height_start, double height_inc, double range_inc);

static struct ionosphere_struct ionosphere;
static int iono_exist_in_mem = 0;

static PyObject *raytrace_2d(PyObject *self, PyObject *args)
{
  char message[256];

  Py_ssize_t num_args = PyTuple_Size(args);

  ASSERT((num_args == 8 || num_args == 9 || num_args == 15 || num_args == 16),
         PyExc_ValueError, "incorrect number of input arguments");

  int init_ionosphere = 1;

  if (num_args == 8 || num_args == 9)
  {
    ASSERT(iono_exist_in_mem, PyExc_RuntimeError,
           "the ionosphere has not been initialized");

    init_ionosphere = 0;
  }

  /*
   * Load input parameters.
   */
  double latitude, longitude, bearing, height_start, height_inc, range_inc;
  double *ray_data;
  double *ray_path_data;
  int nhops, irreg_flags;
  PyObject *in_tol, *input_ray_state;
  PyArrayObject *elevs, *freqs, *iono_en_grid, *iono_en_grid_5, *collision_grid,
      *irreg_grid;

  ASSERT_NOMSG(PyArg_ParseTuple(args, "ddO!dO!iOi|O!O!O!dddO!O!", &latitude, &longitude,
                                &PyArray_Type, &elevs, &bearing, &PyArray_Type, &freqs, &nhops, &in_tol, &irreg_flags,
                                &PyArray_Type, &iono_en_grid, &PyArray_Type, &iono_en_grid_5, &PyArray_Type, &collision_grid,
                                &height_start, &height_inc, &range_inc, &PyArray_Type, &irreg_grid, &PyDict_Type,
                                &input_ray_state));

  npy_intp *elevs_shape = PyArray_DIMS(elevs);

  /* Ensure that `input_ray_state` is the correct size (if supplied). */
  if (num_args == 9 || num_args == 16)
  {
    ASSERT((PyDict_Size(input_ray_state) == 9), PyExc_ValueError,
           "incorrect number of fields in dictionary.");

    PyObject *items = PyDict_Values(input_ray_state);

    for (int i = 0; i < 9; i++)
    {
      PyObject *item = PyList_GetItem(items, i);

      ASSERT(PyArray_Check(item), PyExc_ValueError,
             "field value must be a numpy array");
      ASSERT((PyArray_NDIM((PyArrayObject *)item) == 1), PyExc_ValueError,
             "invalid shape for field");

      npy_intp *item_shape = PyArray_DIMS((PyArrayObject *)item);

      ASSERT((item_shape[0] == elevs_shape[0]), PyExc_ValueError,
             "invalid size for field");
    }
  }

  /* Ensure that `elevs` and `freqs` are the same (and correct) size. */
  ASSERT((PyArray_NDIM(elevs) == 1), PyExc_ValueError,
         "invalid shape for elevs");
  ASSERT((PyArray_NDIM(freqs) == 1), PyExc_ValueError,
         "invalid shape for freqs");

  npy_intp *freqs_shape = PyArray_DIMS(elevs);

  ASSERT((elevs_shape[0] == freqs_shape[0]), PyExc_ValueError,
         "shape of elevs and freqs must be identical");

  /* Ensure the ionosphere grids are valid. */
  if (init_ionosphere)
  {
    verifyIonoGrid(iono_en_grid, iono_en_grid_5, collision_grid, irreg_grid);
  }

  /* Ensure that `nhops` is valid (0 < nhops <= 50). */
  ASSERT((nhops > 0 && nhops <= 50), PyExc_ValueError,
         "number of hops is invalid; must be within the range of 1 through 50");

  /* Ensure that `tol` is valid (can be an integer or list of 3 elements. */
  ASSERT((
             (PyList_CheckExact(in_tol) && PyList_Size(in_tol) == 3) ||
             PyLong_CheckExact(in_tol) ||
             PyFloat_CheckExact(in_tol)),
         PyExc_ValueError, "tol is an invalid shape or type");

  /* Load `elevs` and `freqs`. */
  double *elevs_data = (double *)malloc(elevs_shape[0] * sizeof(double));
  double *freqs_data = (double *)malloc(elevs_shape[0] * sizeof(double));

  npy_intp elevs_stride = PyArray_STRIDE(elevs, 0);
  npy_intp freqs_stride = PyArray_STRIDE(freqs, 0);

  void *elevs_ptr = PyArray_DATA(elevs);
  void *freqs_ptr = PyArray_DATA(freqs);
  double *in_data = (double *)PyArray_DATA(elevs);

  for (int i = 0; i < elevs_shape[0]; i++)
  {
    elevs_data[i] = (*(double *)(elevs_ptr + (i * elevs_stride)));
    freqs_data[i] = *(double *)(freqs_ptr + (i * freqs_stride));
  }

  /* Parse `in_tol`. */
  double tol;
  double step_size_min;
  double step_size_max;

  ASSERT_NOMSG(parse_tol(in_tol, &tol, &step_size_min, &step_size_max));
  /* Load ionosphere */
  if (init_ionosphere)
  {
    buidlIonoStruct(iono_en_grid, iono_en_grid_5, collision_grid,
                    irreg_grid, height_start, height_inc, range_inc);
    iono_exist_in_mem = 1;
  }

  int num_rays = elevs_shape[0];

  /* Parse the `ray_state_vec`. */
  const char *ray_state_fields[] = {"r", "Q", "theta", "delta_r",
                                    "delta_Q", "absorption", "phase_path", "group_path", "group_path_step_size"};

  double *ray_state_vec_in = (double *)malloc(9 * num_rays * sizeof(double));

  if (num_args == 9 || num_args == 16)
  {
    for (int field = 0; field < 9; field++)
    {
      PyObject *val = PyDict_GetItemString(
          input_ray_state,
          ray_state_fields[field]);
      sprintf(
          message,
          "the field \"%s\" is missing from input_ray_state.",
          ray_state_fields[field]);

      ASSERT((val != NULL), PyExc_ValueError, message);

      sprintf(
          message,
          "the field \"%s\" must be a NumPy array.",
          ray_state_fields[field]);

      ASSERT(PyArray_Check(val), PyExc_ValueError, message);

      sprintf(
          message,
          "the field \"%s\" is the incorrect shape.",
          ray_state_fields[field]);

      PyArrayObject *arr = (PyArrayObject *)val;
      arr = (PyArrayObject *)PyArray_Cast(arr, NPY_DOUBLE);

      ASSERT((PyArray_NDIM(arr) == 1), PyExc_ValueError, message);
      ASSERT((PyArray_DIMS(arr)[0] >= num_rays), PyExc_ValueError, message);

      for (npy_intp i = 0; i < num_rays; i++)
      {
        ray_state_vec_in[(i * 9) + field] = *(double *)PyArray_GETPTR1(arr, i);
      }
    }
  }
  else
  {
    for (int i = 0; i < num_rays; i++)
    {
      for (int j = 0; j < 9; j++)
      {
        ray_state_vec_in[(i * 9) + j] = -1;
      }
    }
  }

  int ray_data_size = 9 * MAX_POINTS_IN_RAY * num_rays;

  int *nhops_attempted = (int *)malloc(num_rays * sizeof(int));
  int *npts_in_ray = (int *)malloc(num_rays * sizeof(int));
  ray_data = (double *)malloc(19 * nhops * num_rays * sizeof(double));
  ray_path_data = (double *)malloc(9 * MAX_POINTS_IN_RAY * num_rays * sizeof(double));

  int *ray_label = (int *)malloc(nhops * num_rays * sizeof(int));
  double *ray_state_vec_out = (double *)malloc(ray_data_size * sizeof(double));

  double elapsed_time;

  /* In python, always return the path data and state. */
  int return_ray_path_data = 1;
  int return_ray_state_vec = 1;

  /* Call raytrace_2d_. */
  raytrace_2d_(&latitude, &longitude, &num_rays, elevs_data, &bearing,
               freqs_data, &nhops, &step_size_min, &step_size_max, &tol, &ionosphere,
               &irreg_flags, &return_ray_path_data, &return_ray_state_vec, ray_data,
               ray_path_data, ray_label, nhops_attempted, ray_state_vec_in, npts_in_ray,
               ray_state_vec_out, &elapsed_time);

  PyObject *result = buildOutput(num_rays, nhops_attempted, npts_in_ray, ray_data,
                                 ray_path_data, freqs_data, elevs_data, ray_label, ray_state_vec_out);

  free(elevs_data);
  free(freqs_data);
  free(nhops_attempted);
  free(npts_in_ray);
  free(ray_data);
  free(ray_path_data);
  free(ray_label);
  free(ray_state_vec_in);
  free(ray_state_vec_out);
  return result;
}

static void verifyIonoGrid(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                           PyArrayObject *irreg_grid)
{

  ASSERT((PyArray_NDIM(iono_en_grid) == 2), PyExc_ValueError,
         "invalid shape for iono_en_grid");
  ASSERT((PyArray_NDIM(iono_en_grid_5) == 2), PyExc_ValueError,
         "invalid shape for iono_en_grid_5");
  ASSERT((PyArray_NDIM(collision_grid) == 2), PyExc_ValueError,
         "invalid shape for collision_grid");
  ASSERT((PyArray_NDIM(irreg_grid) == 2), PyExc_ValueError,
         "invalid shape for irreg_grid");

  npy_intp *iono_en_grid_shape = PyArray_DIMS(iono_en_grid);
  npy_intp *iono_en_grid_5_shape = PyArray_DIMS(iono_en_grid_5);
  npy_intp *collision_grid_shape = PyArray_DIMS(collision_grid);
  npy_intp *irreg_grid_shape = PyArray_DIMS(irreg_grid);

  ASSERT((
             iono_en_grid_shape[0] <= max_num_ht &&
             iono_en_grid_shape[1] <= max_num_rng),
         PyExc_ValueError, "iono_en_grid is too large");
  ASSERT((
             iono_en_grid_5_shape[0] <= max_num_ht &&
             iono_en_grid_5_shape[1] <= max_num_rng),
         PyExc_ValueError, "iono_en_grid_5 is too large");
  ASSERT((
             collision_grid_shape[0] <= max_num_ht &&
             collision_grid_shape[1] <= max_num_rng),
         PyExc_ValueError, "collision_grid is too large");
  ASSERT((
             irreg_grid_shape[0] == 4 &&
             irreg_grid_shape[1] <= max_num_rng),
         PyExc_ValueError, "irreg_grid is not the correct size");

  ASSERT((
             iono_en_grid_shape[0] == iono_en_grid_5_shape[0] &&
             iono_en_grid_shape[0] == collision_grid_shape[0]),
         PyExc_ValueError, "ionosphere grids have inconsistent row counts");
  ASSERT((
             iono_en_grid_shape[1] == iono_en_grid_5_shape[1] &&
             iono_en_grid_shape[1] == collision_grid_shape[1] &&
             iono_en_grid_shape[1] == irreg_grid_shape[1]),
         PyExc_ValueError, "ionosphere grids have inconsistent column counts");
}

static PyObject *buildOutput(int num_rays, int *nhops_attempted, int *npts_in_ray, double *ray_data,
                             double *ray_path_data, double *freqs_data, double *elevs_data, int *ray_label, double *ray_state_vec_out)
{
  /* Build output. */
  PyObject *tmp;
  double *tmp_data;

  int rays_num_fields = 23;
  const char *rays_fields[] = {"lat", "lon", "ground_range", "group_range",
                               "phase_path", "geometric_path_length", "initial_elev", "final_elev",
                               "apogee", "gnd_rng_to_apogee", "plasma_freq_at_apogee", "virtual_height",
                               "effective_range", "total_absorption", "deviative_absorption", "TEC_path",
                               "Doppler_shift", "Doppler_spread", "FAI_backscatter_loss", "frequency",
                               "nhops_attempted", "ray_label", "NRT_elapsed_time"};
  int rays_positions[] = {0, 1, 2, 3, 16, 10, 6, 7, 4, 5, 13, 15, 11, 18,
                          12, 17, 9, 8, 14};

  int ray_paths_num_fields = 11;
  const char *ray_paths_fields[] = {"initial_elev", "frequency",
                                    "ground_range", "height", "group_range", "phase_path",
                                    "geometric_distance", "electron_density", "refractive_index",
                                    "collision_frequency", "absorption"};

  int ray_states_num_fields = 9;
  const char *ray_states_fields[] = {"r", "Q", "theta", "delta_r", "delta_Q",
                                     "deviative_absorption", "phase_path", "group_path", "group_step_size"};

  PyObject *py_rays = PyList_New(num_rays);
  PyObject *py_ray_paths = PyList_New(num_rays);
  PyObject *py_ray_states = PyList_New(num_rays);

  for (int ray_id = 0; ray_id < num_rays; ray_id++)
  {
    npy_intp nhops_dims[1] = {nhops_attempted[ray_id]};
    npy_intp npts_dims[1] = {npts_in_ray[ray_id]};

    /* Ray */
    PyObject *py_ray_data = PyDict_New();
    for (int field_id = 0; field_id < rays_num_fields - 4; field_id++)
    {
      int idx = ray_id + (num_rays * rays_positions[field_id]);
      tmp = PyArray_ZEROS(1, nhops_dims, NPY_DOUBLE, 0);
      tmp_data = PyArray_DATA((PyArrayObject *)tmp);

      stepmemcpyd(tmp_data, &ray_data[idx], num_rays * 9,
                  nhops_attempted[ray_id]);
      PyDict_SetItemString(py_ray_data, rays_fields[field_id], tmp);
    }

    PyDict_SetItemString(py_ray_data, "frequency",
                         PyFloat_FromDouble(freqs_data[ray_id]));
    PyDict_SetItemString(py_ray_data, "nhops_attempted",
                         PyLong_FromLong(nhops_attempted[ray_id]));

    tmp = PyArray_ZEROS(1, nhops_dims, NPY_DOUBLE, 0);
    tmp_data = PyArray_DATA((PyArrayObject *)tmp);

    for (int hop_id = 0; hop_id < nhops_attempted[ray_id]; hop_id++)
    {
      *(tmp_data + hop_id) = (double)ray_label[ray_id + num_rays * hop_id];
    }

    PyDict_SetItemString(py_ray_data, "ray_label", tmp);

    /* Ray Path */
    PyObject *py_ray_path_data = PyDict_New();

    for (int field_id = 0; field_id < ray_paths_num_fields - 2; field_id++)
    {
      tmp = PyArray_ZEROS(1, npts_dims, NPY_DOUBLE, 0);

      tmp_data = PyArray_DATA((PyArrayObject *)tmp);

      stepmemcpyd(tmp_data, &ray_path_data[ray_id + (num_rays * field_id)],
                  num_rays * 9, npts_in_ray[ray_id]);

      PyDict_SetItemString(py_ray_path_data, ray_paths_fields[field_id + 2], tmp);
    }

    PyDict_SetItemString(py_ray_path_data, "initial_elev",
                         PyFloat_FromDouble(elevs_data[ray_id]));
    PyDict_SetItemString(py_ray_path_data, "frequency",
                         PyFloat_FromDouble(freqs_data[ray_id]));

    /* Ray State */
    PyObject *py_ray_state_data = PyDict_New();

    for (int field_id = 0; field_id < ray_states_num_fields; field_id++)
    {
      tmp = PyArray_ZEROS(1, npts_dims, NPY_DOUBLE, 0);
      tmp_data = PyArray_DATA((PyArrayObject *)tmp);
      stepmemcpyd(tmp_data, &ray_state_vec_out[ray_id + (num_rays * field_id)],
                  num_rays * 9, npts_in_ray[ray_id]);

      PyDict_SetItemString(py_ray_state_data, ray_states_fields[field_id], tmp);
    }

    PyList_SetItem(py_rays, ray_id, py_ray_data);
    PyList_SetItem(py_ray_paths, ray_id, py_ray_path_data);
    PyList_SetItem(py_ray_states, ray_id, py_ray_state_data);
  }
  return PyTuple_Pack(3, py_rays, py_ray_paths, py_ray_states);
}

static void buidlIonoStruct(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                            PyArrayObject *irreg_grid, double height_start, double height_inc, double range_inc)
{

  /* We can safely use the shape of `iono_en_grid` to represent all the grids
     * because we already checked that their shapes are identical earlier on. */
  npy_intp *grid_shape = PyArray_DIMS(iono_en_grid);

  PyArrayObject *iono_grid = (PyArrayObject *)PyArray_Cast(
      iono_en_grid,
      NPY_DOUBLE);

  PyArrayObject *iono_grid_5 = (PyArrayObject *)PyArray_Cast(
      iono_en_grid_5,
      NPY_DOUBLE);

  PyArrayObject *col_freq = (PyArrayObject *)PyArray_Cast(
      collision_grid,
      NPY_DOUBLE);

  PyArrayObject *irregs = (PyArrayObject *)PyArray_Cast(
      irreg_grid,
      NPY_DOUBLE);
  void *irregs_ptr = PyArray_DATA(irregs);


  for (int i = 0; i < grid_shape[0]; i++)
  {
    for (int j = 0; j < grid_shape[1]; j++)
    {
      npy_intp pos[2] = {i, j};

      ionosphere.eN[j][i] = (*(double *)PyArray_GetPtr(iono_grid, pos)); //houses electron density
      //fprintf(fp,"%f ", ionosphere.eN[i][j]);
      ionosphere.eN_5[j][i] = (*(double *)PyArray_GetPtr(iono_grid_5, pos)); //iono 5 minutes from now
      ionosphere.col_freq[j][i] = (*(double *)PyArray_GetPtr(col_freq, pos));
    }
    //fprintf(fp,"\n");
    ionosphere.irreg_strength[i] = (*(double *)PyArray_GETPTR2(irregs, 0, i)); //nope vals supposed to be really close to zero
    ionosphere.irreg_sma_dip[i] = (*(double *)PyArray_GETPTR2(irregs, 1, i));  //good
    ionosphere.irreg_sma_azim[i] = (*(double *)PyArray_GETPTR2(irregs, 2, i)); //good to go
    ionosphere.dop_spread_sq[i] = (*(double *)PyArray_GETPTR2(irregs, 3, i));  //nope vals supposed to be really close to zero
  }

  //fclose(fp);
  ionosphere.nRange = grid_shape[0]; //num of columns in ion_en-grid_5
  ionosphere.NumHt = grid_shape[1];  //num of rows in iono_en_grid_5
  ionosphere.HtMin = height_start;   //range_inc
  ionosphere.HtInc = height_inc;
  ionosphere.dRange = range_inc;
}

static PyMethodDef methods[] = {
    {"raytrace_2d", raytrace_2d, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "raytrace_2d",
    "",
    -1,
    methods};

PyMODINIT_FUNC PyInit_raytrace_2d()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred())
    return NULL;
  import_array();
  return m;
}
