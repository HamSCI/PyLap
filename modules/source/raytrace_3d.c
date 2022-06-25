#include "../include/pharlap.h"
#include <stdlib.h>
#include <iono_structures_3d.h>

#define MAX_POINTS_IN_RAY 20000

extern void raytrace_3d_(double *start_lat, double *start_long, double *start_height,
                         int *num_rays, double *elevs, double *bearings,
                         double *freqs, int *OX_mode, int *nhops, double *step_size_min,
                         double *step_size_max, double *tol, struct ionosphere_struct *ionosphere,
                         struct geomag_field_struct *geomag_field,
                         double *ray_state_vec_in, int *return_ray_path_data,
                         int *return_ray_state_vec, double *ray_data,
                         double *ray_path_data, int *ray_label, int *nhops_attempted,
                         int *npts_in_ray, double *ray_state_vec_out,
                         double *elapsed_time);

static PyObject *buildOutput(int num_rays, int *nhops_attempted, int *npts_in_ray, double *ray_data, double *bearings_data,
                             double *ray_path_data, double *freqs_data, double *elevs_data, int *ray_label, double *ray_state_vec_out);
static void buildIonoStruct(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                            PyArrayObject *Bx, PyArrayObject *By, PyArrayObject *Bz,
                            PyObject *iono_grid_parms, PyObject *geomag_grid_parms);
static void verifyIono(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                       PyArrayObject *Bx, PyArrayObject *By, PyArrayObject *Bz,
                       PyObject *iono_grid_parms, PyObject *geomag_grid_parms);
void stepmemcpyd(double *dest, double *src, int step, int num_vals);

/* static struct ionosphere_struct ionosphere;  - ionospheric data too large and
   so this statement was causing the stack to be "squashed" ie corrupted - so we
   want to allocate from heap instead */
static struct ionosphere_struct *ptr_ionosphere = NULL;
static struct geomag_field_struct geomag_field;
static int iono_exist_in_mem = 0;

/* clear the ionosphere from memory when mexAtExit is triggered by a clear 
   command at the Matlab prompt */
static void clear_ionosphere(void)
{
  if (ptr_ionosphere != NULL)
  {
    free(ptr_ionosphere);
    ptr_ionosphere = NULL;
  }
}

PyObject *raytrace_3d(PyObject *self, PyObject *args)
{
  char message[256];

  Py_ssize_t num_args = PyTuple_Size(args);

  /* Check for proper number of input arguments. */
  ASSERT((num_args == 9 || num_args == 10 || num_args == 17 || num_args == 18),
         PyExc_ValueError, "incorrect number of input arguments");

  /* determine if ionospheric and geomagnetic grids need to be initialized */
  if (ptr_ionosphere == NULL)
  {
    atexit(clear_ionosphere);
    ptr_ionosphere = malloc(sizeof(*ptr_ionosphere));
  }
  int init_ionosphere = 1;

  if (num_args == 9 || num_args == 10)
  {
    /* OK since we have (nrhs == 9 || nrhs == 10) then an ionosphere has not
       been passed in. Thus, there is not an ionosphere to read in and so we 
       need to set initialize_iono = 0. However, before we do this, check to 
       make sure an ionosphere already exists in memory (i.e. a previous
       raytrace_3d call has already passed it in. If not return an error */
    ASSERT(iono_exist_in_mem, PyExc_RuntimeError,
           "the ionosphere has not been initialized");

    init_ionosphere = 0;
  }

  /*
   * Load input/output parameters.
   */
  double start_lat, start_lon, start_height,
      height_start, height_inc, range_inc, elapsed_time;
  double *ray_data;
  double *ray_path_data;
  int return_ray_path_data;
  int return_ray_state_vec;
  int nhops, OX_mode, num_rays;
  PyObject *input_ray_state_vec, *in_tol, *geomag_grid_parms, *iono_grid_parms;
  PyArrayObject *elevs, *freqs, *bearing, *iono_en_grid, *iono_en_grid_5, *collision_grid,
      *By, *Bx, *Bz;

  // (origin_lat, origin_long, origin_ht, elevs, ray_bears, freqs,
  //             OX_mode, nhops, tol, iono_en_grid, iono_en_grid_5,
  //           collision_freq, iono_grid_parms, Bx, By, Bz,
  //           geomag_grid_parms)
  ASSERT_NOMSG(PyArg_ParseTuple(args, "dddO!O!O!iiO|O!O!O!OO!O!O!OO!", &start_lat, &start_lon,
                                &start_height, &PyArray_Type, &elevs, &PyArray_Type, &bearing, &PyArray_Type, &freqs, &OX_mode,
                                &nhops, &in_tol, &PyArray_Type, &iono_en_grid, &PyArray_Type, &iono_en_grid_5,
                                &PyArray_Type, &collision_grid, &iono_grid_parms,
                                &PyArray_Type, &Bx, &PyArray_Type, &By, &PyArray_Type, &Bz, &geomag_grid_parms,
                                &PyDict_Type, &input_ray_state_vec));

  npy_intp *elevs_shape = PyArray_DIMS(elevs);

  /* Ensure that `tol` is valid (can be an integer or list of 3 elements. */
  ASSERT((
             (PyList_CheckExact(in_tol) && PyList_Size(in_tol) == 3) ||
             PyLong_CheckExact(in_tol) ||
             PyFloat_CheckExact(in_tol)),
         PyExc_ValueError, "tol is an invalid shape or type");

  if (init_ionosphere)
  {
    verifyIono(iono_en_grid, iono_en_grid_5, collision_grid,
               Bx, By, Bz,
               iono_grid_parms, geomag_grid_parms);
  }

  /* Ensure that `start_lat` is valid (-90 < nhops <= 90). */
  ASSERT((start_lat >= -90.0 && start_lat <= 90.0), PyExc_ValueError,
         "start_lat is invalid; must be within the range of -90 through 90");

  /* Ensure that `start_lon` is valid (-180 < nhops <= 180). */
  ASSERT((start_lon >= -180.0 && start_lon <= 180.0), PyExc_ValueError,
         "start_lon is invalid; must be within the range of -180 through 180");

  /* Ensure that `nhops` is valid (0 < nhops <= 50). */
  ASSERT((nhops > 0 && nhops <= 50), PyExc_ValueError,
         "number of hops is invalid; must be within the range of 1 through 50");

  /* Load `elevs` and `freqs`. */
  double *elevs_data = (double *)malloc(elevs_shape[0] * sizeof(double));
  double *freqs_data = (double *)malloc(elevs_shape[0] * sizeof(double));
  double *bearings_data = (double *)malloc(elevs_shape[0] * sizeof(double));

  npy_intp elevs_stride = PyArray_STRIDE(elevs, 0);
  npy_intp freqs_stride = PyArray_STRIDE(freqs, 0);
  npy_intp bearings_stride = PyArray_STRIDE(bearing, 0);

  void *elevs_ptr = PyArray_DATA(elevs);
  void *freqs_ptr = PyArray_DATA(freqs);
  void *bearings_ptr = PyArray_DATA(bearing);

  for (int i = 0; i < elevs_shape[0]; i++)
  {
    elevs_data[i] = (*(double *)(elevs_ptr + (i * elevs_stride)));
    freqs_data[i] = *(double *)(freqs_ptr + (i * freqs_stride));
    bearings_data[i] = *(double *)(bearings_ptr + (i * bearings_stride));
  }

  /* Parse `in_tol`. */
  double tol;
  double step_size_min;
  double step_size_max;

  ASSERT_NOMSG(parse_tol(in_tol, &tol, &step_size_min, &step_size_max));

  if (init_ionosphere)
  {

    buildIonoStruct(iono_en_grid, iono_en_grid_5, collision_grid,
                    Bx, By, Bz,
                    iono_grid_parms, geomag_grid_parms);
    /* Now the ionosphere has been read in set the iono_exist_in_mem flag to
       indicate this for future raytrace calls */
    iono_exist_in_mem = 1;
  }
  num_rays = elevs_shape[0];
  /* If we are doing a magneto-ionic raytrace then check that the magnetic
     field grid is consistent with the ionospheric grid */
  if (OX_mode != 0)
  {

    ASSERT((geomag_field.lat_min == ptr_ionosphere->lat_min), PyExc_ValueError,
           "\nAThe minimum latitude of the geomagnetic field grid (deg.) is inconsistent with the electron density grid ( deg).");

    ASSERT((geomag_field.lat_max == ptr_ionosphere->lat_max), PyExc_ValueError,
           "\nAThe maximum latitude of the geomagnetic field grid (deg.) is inconsistent with the electron density grid (deg).");

    ASSERT((geomag_field.lon_min == ptr_ionosphere->lon_min), PyExc_ValueError,
           "\nBThe minimum longitude of the geomagnetic field grid (deg.) is inconsistent with the electron density grid (deg).");

    ASSERT((geomag_field.lon_max == ptr_ionosphere->lon_max), PyExc_ValueError,
           "\nBThe maximum longitude of the geomagnetic field grid (deg.) is inconsistent with the electron density grid (deg).");

    ASSERT((geomag_field.ht_min == ptr_ionosphere->ht_min), PyExc_ValueError,
           "\nCThe minimum height of the geomagnetic field grid ( deg.) is inconsistent with the electron density grid ( deg).");

    ASSERT((geomag_field.ht_max == ptr_ionosphere->ht_max), PyExc_ValueError,
           "\nCThe maximum height of the geomagnetic field grid (deg.) is inconsistent with the electron density grid (deg).");
  }
  /* If a user specified input state vector is not being used then check to make
     sure that start height of raytracing is below the start of the ionosphere.
     If not then quit with error message. */
  if (num_args != 10 && num_args != 18)
  {
    ASSERT((start_height <= ptr_ionosphere->ht_min), PyExc_ValueError,
           "The start height for ray tracing must be below the start height of the ionospheric grid. If you want to start ray tracing inside the ionosphere then you must also specify the initial state vector of the ray");
  }

  /* read in the optional input (structure containing a user defined starting 
     ray state vector for each ray) from  Matlab (if required) and check to make
     sure each field is valid */
  const char *ray_state_fields[] = {"pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "group_path",
                                    "geometrical_path", "phase_path", "absorption", "indep_var", "ODE_step_size"};

  double *ray_state_vec_in = (double *)malloc(12 * num_rays * sizeof(double));

  if (num_args == 10 || num_args == 18)
  {
    for (int field = 0; field < 12; field++)
    {
      PyObject *val = PyDict_GetItemString(
          input_ray_state_vec,
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
        ray_state_vec_in[(i * 12) + field] = *(double *)PyArray_GETPTR1(arr, i);
      }
    }
  }
  else
  {
    for (int i = 0; i < num_rays; i++)
    {
      for (int j = 0; j < 12; j++)
      {
        ray_state_vec_in[(i * 12) + j] = -1;
      }
    }
  }

  /* init pointers and malloc space for return data */
  int *nhops_attempted = (int *)malloc(num_rays * sizeof(int));
  int *npts_in_ray = (int *)malloc(num_rays * sizeof(int));
  ray_data = (double *)malloc(15 * nhops * num_rays * sizeof(double));
  ray_path_data = (double *)malloc(19 * MAX_POINTS_IN_RAY * num_rays * sizeof(double));
  int *ray_label = (int *)malloc(nhops * num_rays * sizeof(int));
  double *ray_state_vec_out = (double *)malloc(12 * MAX_POINTS_IN_RAY * num_rays * sizeof(double));

  /* determine if the ray_path_data and ray_state_vec arrays have been 
     requested to be returned to matlab */
  return_ray_path_data = 1; /* default action - don't return this array */
  return_ray_state_vec = 1; /* default action - don't return this array */

  /* call the computational routine raytrace_3d_ */
  step_size_min = step_size_min * 1000.0; /* convert to meters */
  step_size_max = step_size_max * 1000.0; /* convert to meters */

  raytrace_3d_(&start_lat, &start_lon, &start_height, &num_rays, elevs_data,
               bearings_data, freqs_data, &OX_mode, &nhops, &step_size_min,
               &step_size_max, &tol, ptr_ionosphere, &geomag_field,
               ray_state_vec_in, &return_ray_path_data, &return_ray_state_vec,
               ray_data, ray_path_data, ray_label, nhops_attempted, npts_in_ray,
               ray_state_vec_out, &elapsed_time);

  PyObject *result = buildOutput(num_rays, nhops_attempted, npts_in_ray, ray_data, bearings_data,
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

static PyObject *buildOutput(int num_rays, int *nhops_attempted, int *npts_in_ray, double *ray_data, double *bearings_data,
                             double *ray_path_data, double *freqs_data, double *elevs_data, int *ray_label, double *ray_state_vec_out)
{

  const int ray_data_numfields = 19;
  const char *ray_data_fieldnames[] =
      {"lat", "lon", "ground_range", "group_range", "phase_path",
       "initial_elev", "final_elev", "initial_bearing", "final_bearing",
       "total_absorption", "deviative_absorption", "TEC_path", "Doppler_shift",
       "apogee", "geometric_path_length", "frequency", "nhops_attempted",
       "ray_label", "NRT_elapsed_time"};
  const int ray_data_fieldname_raytrace_output_position[] =
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  const int ray_path_data_numfields = 22;
  const char *ray_path_data_fieldnames[] =
      {"initial_elev", "initial_bearing", "frequency", "lat", "lon", "height",
       "group_range", "phase_path", "refractive_index", "group_refractive_index",
       "wavenorm_ray_angle", "wavenorm_B_angle", "polariz_mag",
       "wave_Efield_tilt", "volume_polariz_tilt", "electron_density", "geomag_x",
       "geomag_y", "geomag_z", "geometric_distance", "collision_frequency",
       "absorption"};
  const int ray_state_vec_numfields = 12;
  const char *ray_state_vec_fieldnames[] =
      {"pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "group_path",
       "geometrical_path", "phase_path", "absorption", "indep_var", "ODE_step_size"};
  /*build output Copy the raytrace data into the py data structures. */
  PyObject *tmp;
  double *tmp_data;
  PyObject *py_rays = PyList_New(num_rays);
  PyObject *py_ray_paths = PyList_New(num_rays);
  PyObject *py_ray_states = PyList_New(num_rays);

  for (int ray_id = 0; ray_id < num_rays; ray_id++)
  {
    npy_intp nhops_dims[1] = {nhops_attempted[ray_id]};
    npy_intp npts_dims[1] = {npts_in_ray[ray_id]};

    /* Ray */
    PyObject *py_ray_data = PyDict_New();

    for (int field_id = 0; field_id < ray_data_numfields - 4; field_id++)
    {
      int idx = ray_id + (num_rays * ray_data_fieldname_raytrace_output_position[field_id]);
      tmp = PyArray_ZEROS(1, nhops_dims, NPY_DOUBLE, 0);
      tmp_data = PyArray_DATA((PyArrayObject *)tmp);
      stepmemcpyd(tmp_data, &ray_data[idx], num_rays * 15,
                  nhops_attempted[ray_id]);
      PyDict_SetItemString(py_ray_data, ray_data_fieldnames[field_id], tmp);
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

    for (int field_id = 0; field_id < ray_path_data_numfields - 3; field_id++)
    {
      tmp = PyArray_ZEROS(1, npts_dims, NPY_DOUBLE, 0);

      tmp_data = PyArray_DATA((PyArrayObject *)tmp);

      stepmemcpyd(tmp_data, &ray_path_data[ray_id + (num_rays * field_id)],
                  num_rays * 19, npts_in_ray[ray_id]);

      PyDict_SetItemString(py_ray_path_data, ray_path_data_fieldnames[field_id + 3], tmp);
    }

    PyDict_SetItemString(py_ray_path_data, "initial_elev",
                         PyFloat_FromDouble(elevs_data[ray_id]));
    PyDict_SetItemString(py_ray_path_data, "initial_bearing",
                         PyFloat_FromDouble(bearings_data[ray_id]));
    PyDict_SetItemString(py_ray_path_data, "frequency",
                         PyFloat_FromDouble(freqs_data[ray_id]));

    /* Ray State */
    PyObject *py_ray_state_data = PyDict_New();

    for (int field_id = 0; field_id < ray_state_vec_numfields; field_id++)
    {
      tmp = PyArray_ZEROS(1, npts_dims, NPY_DOUBLE, 0);
      tmp_data = PyArray_DATA((PyArrayObject *)tmp);
      stepmemcpyd(tmp_data, &ray_state_vec_out[ray_id + (num_rays * field_id)],
                  num_rays * 12, npts_in_ray[ray_id]);

      PyDict_SetItemString(py_ray_state_data, ray_state_vec_fieldnames[field_id], tmp);
    }

    PyList_SetItem(py_rays, ray_id, py_ray_data);
    PyList_SetItem(py_ray_paths, ray_id, py_ray_path_data);
    PyList_SetItem(py_ray_states, ray_id, py_ray_state_data);
  }
  return PyTuple_Pack(3, py_rays, py_ray_paths, py_ray_states);
}

static void verifyIono(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                       PyArrayObject *Bx, PyArrayObject *By, PyArrayObject *Bz,
                       PyObject *iono_grid_parms, PyObject *geomag_grid_parms)
{
  ASSERT((PyArray_NDIM(iono_en_grid) == 3), PyExc_ValueError,
         "invalid shape for iono_en_grid");
  ASSERT((PyArray_NDIM(iono_en_grid_5) == 3), PyExc_ValueError,
         "invalid shape for iono_en_grid_5");
  ASSERT((PyArray_NDIM(collision_grid) == 3), PyExc_ValueError,
         "invalid shape for collision_grid");

  npy_intp *iono_en_grid_shape = PyArray_DIMS(iono_en_grid);
  npy_intp *iono_en_grid_5_shape = PyArray_DIMS(iono_en_grid_5);
  npy_intp *collision_grid_shape = PyArray_DIMS(collision_grid);

  ASSERT((
             iono_en_grid_shape[0] <= max_num_lat &&
             iono_en_grid_shape[1] <= max_num_lon &&
             iono_en_grid_shape[2] <= max_num_ht),
         PyExc_ValueError, "iono_en_grid is too large");
  ASSERT((
             iono_en_grid_5_shape[0] <= max_num_lat &&
             iono_en_grid_5_shape[1] <= max_num_lon &&
             iono_en_grid_5_shape[2] <= max_num_ht),
         PyExc_ValueError, "iono_en_grid_5 is too large");
  ASSERT((
             collision_grid_shape[0] <= max_num_lat &&
             collision_grid_shape[1] <= max_num_lon &&
             collision_grid_shape[2] <= max_num_ht),
         PyExc_ValueError, "collision_grid is too large");

  ASSERT((
             iono_en_grid_shape[0] == iono_en_grid_5_shape[0] &&
             iono_en_grid_shape[0] == collision_grid_shape[0]),
         PyExc_ValueError, "ionosphere grids have inconsistent row counts");
  ASSERT((
             iono_en_grid_shape[1] == iono_en_grid_5_shape[1] &&
             iono_en_grid_shape[1] == collision_grid_shape[1]),
         PyExc_ValueError, "ionosphere grids have inconsistent column counts");

  // check B grid for shape and consistency
  ASSERT((PyArray_NDIM(Bx) == 3), PyExc_ValueError,
         "invalid shape for Bx");
  ASSERT((PyArray_NDIM(By) == 3), PyExc_ValueError,
         "invalid shape for By");
  ASSERT((PyArray_NDIM(By) == 3), PyExc_ValueError,
         "invalid shape for Bz");
  npy_intp *Bx_shape = PyArray_DIMS(Bx);
  npy_intp *By_shape = PyArray_DIMS(By);
  npy_intp *Bz_shape = PyArray_DIMS(Bz);

  ASSERT((
             Bx_shape[0] <= 101 &&
             Bx_shape[1] <= 101 &&
             Bx_shape[2] <= 201),
         PyExc_ValueError, "Bx is too large");
  ASSERT((
             By_shape[0] <= 101 &&
             By_shape[1] <= 101 &&
             By_shape[2] <= 201),
         PyExc_ValueError, "By is too large");
  ASSERT((
             Bz_shape[0] <= 101 &&
             Bz_shape[1] <= 101 &&
             Bz_shape[2] <= 201),
         PyExc_ValueError, "Bz is too large");
  ASSERT((
             Bx_shape[0] == By_shape[0] &&
             By_shape[0] == Bz_shape[0]),
         PyExc_ValueError, "B grids have inconsistent row counts");
  ASSERT((
             Bx_shape[1] == By_shape[1] &&
             By_shape[1] == Bz_shape[1]),
         PyExc_ValueError, "B grids have inconsistent column counts");
  ASSERT((
             Bx_shape[2] == By_shape[2] &&
             By_shape[2] == Bz_shape[2]),
         PyExc_ValueError, "B grids have inconsistent height counts");

  // ensure the parms are valid
  ASSERT((PyList_Size(iono_grid_parms) == 9), PyExc_ValueError,
         "invalid shape for iono_grid_parms");
  ASSERT((PyList_Size(geomag_grid_parms) == 9), PyExc_ValueError,
         "invalid shape for geomag_grid_parms");
}

static void buildIonoStruct(PyArrayObject *iono_en_grid, PyArrayObject *iono_en_grid_5, PyArrayObject *collision_grid,
                            PyArrayObject *Bx, PyArrayObject *By, PyArrayObject *Bz,
                            PyObject *iono_grid_parms, PyObject *geomag_grid_parms)
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

  for (int k = 0; k < grid_shape[2]; k++)
  {
    for (int j = 0; j < grid_shape[1]; j++)
    {
      for (int i = 0; i < grid_shape[0]; i++)
      {
        npy_intp pos[3] = {i, j, k};
        ptr_ionosphere->eN[k][j][i] = (*(double *)PyArray_GetPtr(iono_grid, pos));     //houses electron density
        ptr_ionosphere->eN_5[k][j][i] = (*(double *)PyArray_GetPtr(iono_grid_5, pos)); //iono 5 minutes from now
        ptr_ionosphere->col_freq[k][j][i] = (*(double *)PyArray_GetPtr(col_freq, pos));
      }
    }
  }

  ptr_ionosphere->lat_min = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 0)); //num of columns in ion_en-grid_5
  ptr_ionosphere->lat_inc = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 1)); //num of rows in iono_en_grid_5
  // /* Ensure that `start_lat` is valid (-90 < nhops <= 90). */
  ASSERT((ptr_ionosphere->lat_min >= -90.0 && ptr_ionosphere->lat_min <= 90.0), PyExc_ValueError,
         "lat_min is invalid; must be within the range of -90 through 90");

  ptr_ionosphere->num_lat = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 2)); //range_inc
  /* Ensure that `start_lon` is valid (-180 < nhops <= 180). */
  ASSERT((grid_shape[0] == ptr_ionosphere->num_lat), PyExc_ValueError,
         "grid shape != num_lat");

  ptr_ionosphere->lat_max = ptr_ionosphere->lat_min +
                            (ptr_ionosphere->num_lat - 1) * ptr_ionosphere->lat_inc;

  ptr_ionosphere->lon_min = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 3));
  ASSERT((ptr_ionosphere->lon_min >= -180 && ptr_ionosphere->lon_min <= 180.0), PyExc_ValueError,
         "lon_min must be within -180 and 180");

  ptr_ionosphere->lon_inc = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 4));
  ptr_ionosphere->num_lon = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 5));
  ASSERT((grid_shape[1] == ptr_ionosphere->num_lon), PyExc_ValueError,
         "grid shape[1] != num_lon");

  ptr_ionosphere->lon_max = ptr_ionosphere->lon_min +
                            (ptr_ionosphere->num_lon - 1) * ptr_ionosphere->lon_inc;
  ptr_ionosphere->ht_min = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 6));
  ptr_ionosphere->ht_inc = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 7));
  ptr_ionosphere->num_ht = PyFloat_AsDouble(PyList_GetItem(iono_grid_parms, 8));
  ASSERT((grid_shape[2] == ptr_ionosphere->num_ht), PyExc_ValueError,
         "grid shape[2] != num_ht");

  ptr_ionosphere->ht_max = ptr_ionosphere->ht_min +
                           (ptr_ionosphere->num_ht - 1) * ptr_ionosphere->ht_inc;

  npy_intp *dims = PyArray_DIMS(Bx);

  for (int k = 0; k < dims[2]; k++)
  {
    for (int j = 0; j < dims[1]; j++)
    {
      for (int i = 0; i < dims[0]; i++)
      {
        npy_intp pos[3] = {i, j, k};
        geomag_field.Bx[k][j][i] = (*(double *)PyArray_GetPtr(Bx, pos));
        geomag_field.By[k][j][i] = (*(double *)PyArray_GetPtr(By, pos));
        geomag_field.Bz[k][j][i] = (*(double *)PyArray_GetPtr(Bz, pos));
      }
    }
  }
  geomag_field.lat_min = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 0));
  ASSERT((geomag_field.lat_min >= -90.0 && geomag_field.lat_min <= 90), PyExc_ValueError,
         "The start latitude of the input geomagnetic field grid must be in the range -90 to 90 degrees.");

  geomag_field.lat_inc = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 1));
  geomag_field.num_lat = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 2));
  ASSERT((dims[0] == geomag_field.num_lat), PyExc_ValueError,
         "The number of latitudes in the input geomagnetic field grid does not match the input grid parameter.");

  geomag_field.lat_max = geomag_field.lat_min +
                         (geomag_field.num_lat - 1) * geomag_field.lat_inc;
  geomag_field.lon_min = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 3));

  ASSERT((geomag_field.lon_min >= -180.0 && geomag_field.lon_min <= 180.0), PyExc_ValueError,
         "he start longitude of the input geomagnetic field grid must be in the range -180 to 180 degrees.");

  geomag_field.lon_inc = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 4));
  geomag_field.num_lon = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 5));
  ASSERT((dims[1] == geomag_field.num_lon), PyExc_ValueError,
         "The number of longitudes in the input geomagnetic field grid does not match the input grid parameter.");

  geomag_field.lon_max = geomag_field.lon_min +
                         (geomag_field.num_lon - 1) * geomag_field.lon_inc;
  geomag_field.ht_min = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 6));
  geomag_field.ht_inc = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 7));
  geomag_field.num_ht = PyFloat_AsDouble(PyList_GetItem(geomag_grid_parms, 8));
  ASSERT((dims[2] == geomag_field.num_ht), PyExc_ValueError,
         "The number of heights in the input geomagnetic field grid does not match the input grid parameter.");

  geomag_field.ht_max = geomag_field.ht_min +
                        (geomag_field.num_ht - 1) * geomag_field.ht_inc;
}
static PyMethodDef methods[] = {
    {"raytrace_3d", raytrace_3d, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "raytrace_3d",
    "",
    -1,
    methods};

PyMODINIT_FUNC PyInit_raytrace_3d()
{
  PyObject *m = PyModule_Create(&module);
  if (m == NULL || PyErr_Occurred())
    return NULL;
  import_array();
  return m;
}
