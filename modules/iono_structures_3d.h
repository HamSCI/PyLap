/* 
  Name:
    iono_structures_3d.fin
 
  Purpose:
     ontains the type structures for compiling PHaRLAP
 
  Warning :
    Any changes made to this include file must be followed by a rebuild of
    every module that references this file and programs using those modules.
 
  Modification History:
    29-08-2009  M. A. Cervera
      Initial Version
    
    31-07-2014  M. A. Cervera
      Increased the ionospheric grid sizes from 201x201x401 to 701x701x301 lat, 
      lon, heights.

    22-04-2015  M. A. Cervera
      Increased the ionospheric grid sizes from 701x701x301 to 701x701x401 lat, 
      lon, heights.

    04-05-2015  M. A. Cervera
      Enabled build of Win32 by using #define to specifiy smaller max. iono
      grid sizes for that platform.
*/ 

#if defined WIN32
  #define max_num_ht 301
  #define max_num_lon 501
  #define max_num_lat 501
#elif defined P6
  #define max_num_ht 3001
  #define max_num_lon 201
  #define max_num_lat 201
#else
  #define max_num_ht  401
  #define max_num_lon 701
  #define max_num_lat 701
#endif

/* structure containing ionosphere */
struct ionosphere_struct {
  /* Array of electron densities (number/cm^3) for each height, longitude and 
     latitude height  */
  double eN[max_num_ht][max_num_lon][max_num_lat];       

  /* The electron dens  profiles 5 mins later */
  double eN_5[max_num_ht][max_num_lon][max_num_lat];

  /* Collision frequency array (MHz) */
  double col_freq[max_num_ht][max_num_lon][max_num_lat];
  
  double ht_min;                  /* minimum height for arrays  */ 
  int    num_ht;                  /* number of heights for arrays  */
  double ht_inc;                  /* height increment for arrays  */
  double ht_max;                  /* maximum height for arrays */
  double lat_min;                 /* minimum latitude for arrays  */
  int    num_lat;                 /* number of latitudes for arrays  */
  double lat_inc;                 /* latitude increment for arrays  */
  double lat_max;                 /* maximum latitude for arrays */
  double lon_min;                 /* minium longitude for arrays  */
  int    num_lon;                 /* number of longitudes for arrays  */
  double lon_inc;                 /* longitude increment for arrays  */
  double lon_max;                 /* maximum longitude for arrays */
};

/* struct ionosphere_struct ionosphere; */

/* structure containing geomagnetic field */
struct geomag_field_struct {
  double Bx[201][101][101];    /* Arrays of x, y and z components of the */
  double By[201][101][101];    /*   geomagnetic field (Tesla) for each */
  double Bz[201][101][101];    /*   height, longitude, and latitude */
  double ht_min;               /* minimum height for arrays  */ 
  int    num_ht;               /* number of heights for arrays  */
  double ht_inc;               /* height increment for arrays  */
  double ht_max;               /* maximum height for arrays */
  double lat_min;              /* minium latitude for arrays  */
  int    num_lat;              /* number of latitudes for arrays  */
  double lat_inc;              /* latitude increment for arrays  */
  double lat_max;              /* maximum latitude for arrays */
  double lon_min;              /* minium longitude for arrays  */
  int    num_lon;              /* number of longitudes for arrays  */
  double lon_inc;              /* longitude increment for arrays  */
  double lon_max;              /* maximum longitude for arrays */
};

/* struct geomag_field_struct geomag_field; */
