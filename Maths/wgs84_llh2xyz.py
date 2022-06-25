#M
#M Name:
#M   wgs84_llh2xyz.m
#M
#M Purpose:
#M   Convert WGS84 ellipsoidal geodetic lat, long, and height to geocentric 
#M   x, y, z coordinates.
#M 
#M Calling sequence:
#M   [x, y, z] = wgs84_llh2xyz(geod_lat, geod_long, height)
#M
#M Inputs:
#M   geod_lat  -  array of geodetic latitude (deg)
#M   geod_long -  array of geodetic longitude
#M   height    -  array of heights above ellipsoid (m)              
#M 
#M Outputs: 
#M   x, y, z    -  arrays of geocentric x, y, z (m)
#M
#M Dependencies:
#M   none
#M
#M Modification history:
#M   01/08/2007 M. A. Cervera
#M     Initial version.  
#M
#   06/08/2020 W. C. Liles
#      Convert to Python
#
import numpy as np
import math
#
#
# function [x, y, z] = wgs84_llh2xyz(geod_lat, geod_long, height)
def wgs84_llh2xyz(geod_lat, geod_long, height):
  dtor = math.pi/180.0	        #M Degrees to radians.
  a = 6378137.0                #M WGS84 semimajor axis (m)
  f = 1.0 / 298.257223563     #M WGS84 flattening factor
  b = a * (1 - f)                 #M WGS84 semiminor axis (m)
  e2 = (a ** 2 - b ** 2) / a ** 2   #M eccentricity of ellipsoid squared

  #M Convert geodetic lat, long to radians
  geod_long_r = geod_long * dtor		
  geod_lat_r = geod_lat * dtor
  
  #M do the calculation
  sin_gd_lat = np.sin(geod_lat_r)
  cos_gd_lat = np.cos(geod_lat_r)
  sin_gd_lon = np.sin(geod_long_r)
  cos_gd_lon = np.cos(geod_long_r)

  chi = np.sqrt(1 - e2 * sin_gd_lat ** 2)

  x = (a / chi + height) * cos_gd_lat * cos_gd_lon
  y = (a / chi + height) * cos_gd_lat * sin_gd_lon
  z = ((a * (1 - e2)) / chi + height) * sin_gd_lat

  return x, y, z
  

