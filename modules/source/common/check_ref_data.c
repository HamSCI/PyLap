#include <stdlib.h>
#include <stdio.h>

#define NO_IMPORT_ARRAY
#include "../../include/pharlap.h"


int check_iri2007(const char *);
int check_iri2012(const char *);
int check_iri2016(const char *);
int check_land_sea(const char *);

int check_file_exists(const char *);

int check_ref_data(const char *files_to_check)
{
  char *refdata_dir = getenv("DIR_MODELS_REF_DAT");
  ASSERT_INT((refdata_dir != NULL), PyExc_EnvironmentError,
    "the environment variable \"DIR_MODELS_REF_DAT\" must be defined");

  if (strncmp(files_to_check, "iri2007", 7) == 0) {
    return check_iri2007(refdata_dir);
  } else if (strncmp(files_to_check, "iri2012", 7) == 0) {
    return check_iri2012(refdata_dir);
  } else if (strncmp(files_to_check, "iri2016", 7) == 0) {
    return check_iri2016(refdata_dir);
  } else if (strncmp(files_to_check, "land_sea", 8) == 0) {
    return check_land_sea(refdata_dir);
  }

  PyErr_SetString(PyExc_ValueError, "invalid reference data type.");
  return 0;
}

int check_iri2007(const char *refdata_dir)
{
  char filename[256];

  snprintf(filename, 256, "%s/iri2007/dgrf00.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  for (int idx = 45; idx <= 95; idx += 5) {
    snprintf(filename, 256, "%s/iri2007/dgrf%d.dat", refdata_dir, idx);
    ASSERT_INT_NOMSG(check_file_exists(filename));
  }

  snprintf(filename, 256, "%s/iri2007/igrf05.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  snprintf(filename, 256, "%s/iri2007/igrf05s.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  snprintf(filename, 256, "%s/iri2007/ap.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  snprintf(filename, 256, "%s/iri2007/ig_rz.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  for (int month = 1; month <= 12; month++) {
    snprintf(filename, 256, "%s/iri2007/ccir%d.asc", refdata_dir, month+10);
    ASSERT_INT_NOMSG(check_file_exists(filename));

    snprintf(filename, 256, "%s/iri2007/ursi%d.asc", refdata_dir, month+10);
    ASSERT_INT_NOMSG(check_file_exists(filename));
  }

  return 1;
}

int check_iri2012(const char *refdata_dir)
{
  char filename[256];

  snprintf(filename, 256, "%s/iri2012/igrf2015.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  snprintf(filename, 256, "%s/iri2012/igrf2015s.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  for (int year = 1945; year <= 2010; year += 5) {
    snprintf(filename, 256, "%s/iri2012/dgrf%d.dat", refdata_dir, year);
    ASSERT_INT_NOMSG(check_file_exists(filename));
  }

  snprintf(filename, 256, "%s/iri2012/apf107.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  snprintf(filename, 256, "%s/iri2012/ig_rz.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  for (int month = 1; month <= 12; month++) {
    snprintf(filename, 256, "%s/iri2012/ccir%d.asc", refdata_dir, month+10);
    ASSERT_INT_NOMSG(check_file_exists(filename));

    snprintf(filename, 256, "%s/iri2012/ursi%d.asc", refdata_dir, month+10);
    ASSERT_INT_NOMSG(check_file_exists(filename));
  }

  return 1;
}

int check_iri2016(const char *refdata_dir)
{
  char filename[256];

  snprintf(filename, 256, "%s/iri2016/igrf2015.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  snprintf(filename, 256, "%s/iri2016/igrf2015s.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  for (int year = 1945; year <= 2010; year += 5) {
    snprintf(filename, 256, "%s/iri2016/dgrf%d.dat", refdata_dir, year);
    ASSERT_INT_NOMSG(check_file_exists(filename));
  }

  snprintf(filename, 256, "%s/iri2016/apf107.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  snprintf(filename, 256, "%s/iri2016/ig_rz.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));

  for (int month = 1; month <= 12; month++) {
    snprintf(filename, 256, "%s/iri2016/mcsat%d.dat", refdata_dir, month+10);
    ASSERT_INT_NOMSG(check_file_exists(filename));

    snprintf(filename, 256, "%s/iri2016/ccir%d.asc", refdata_dir, month+10);
    ASSERT_INT_NOMSG(check_file_exists(filename));

    snprintf(filename, 256, "%s/iri2016/ursi%d.asc", refdata_dir, month+10);
    ASSERT_INT_NOMSG(check_file_exists(filename));
  }

  return 1;
}

int check_land_sea(const char *refdata_dir)
{
  char filename[256];
  snprintf(filename, 256, "%s/global_land_Mask_3600_by_1800.dat", refdata_dir);
  ASSERT_INT_NOMSG(check_file_exists(filename));
  return 1;
}

int check_file_exists(const char *filename)
{
  int result = 1;
  FILE *file = fopen(filename, "r");

  if (file == NULL) {
    char message[256];
    snprintf(message, 256, "Reference data file, %s, does not exist or is "
      "unreadable. Check that the environment variable DIR_MODELS_REF_DAT is "
      "set to the directory containing the reference data.", filename);
    PyErr_SetString(PyExc_EnvironmentError, message);
    result = 0;
  } else {
    fclose(file);
  }

  return result;
}
