#M
#M Name:
#M   wgs84_xyz2llh.m
#M
#M Purpose:
#M   Convert geocentric x, y, z coordinates to WGS84 ellipsoidal geodetic lat, 
#M   long, and height.
#M 
#M Calling sequence:
#M   [geod_lat, geod_long, height] = wgs84_xyz2llh(x, y, z)
#M
#M Inputs (scalar):
#M   x, y, z    -  geocentric x,y,z (m)
#M
#M Outputs (scalar):
#M   geod_long, geod_lat -  geodetic longitude, latitude (deg)
#M   height              -  height above ellipsoid (m)              
#M 
#M Modification history:
#M   31/07/2007 M. A. Cervera
#M     Initial version. Converted from IDL code written by  R. Sterner,   
#M     Johns Hopkins University/Applied Physics Laboratory, 2002 May 06. 
#M
#M   09/06/2018 M. A. Cervera
#M     Fixed bug where routine would crash for certain input values
#M
#    07/07/2020  W. C. Liles
#      Convert to Python
#
#
import numpy as np
#
# function [geod_lat, geod_long, height] = wgs84_xyz2llh(x, y, z)
def wgs84_xyz2llh(x, y, z):
  a = 6378137.0               #M WGS84 semimajor axis (m)
  f = 1.0 / 298.257223563    #M WGS84 flattening factor
  b = a * (1 - f)                #M WGS84 semiminor axis (m)

  a2 = a ** 2
  a4 = a ** 4
  b2 = b ** 2
  b4 = b ** 4
  a2b2 = a2 * b2

  #M Convert problem from 3-D to 2-D
  x0 = np.sqrt(x ** 2 + y ** 2)
  y0 = z
  x02 = x0 ** 2
  y02 = y0 ** 2

  #M Coefficients of the polynomial
  c = np.zeros(5)
  c[4] = a2b2 * (b2 * x02 + a2 * y02 - a2b2)
  c[3] = 2 * a2b2 * (x02 + y02 - a2 - b2)
  c[2] = a2 * (x02 - 4.* b2) + b2 * y02 - a4 - b4
  c[1] = -2 * (a2 + b2)
  c[0] = -1.0

  #M Find roots of the 4th degree polynomial.   
  rr = np.roots(c)

  #M Nearest = nadir point.
  t = np.real(rr[3])	

  #M Ellipse X, Y at nadir point.
  xe = a2 * x0 / (a2 + t)	   
  ye = b2 * y0 / (b2 + t)

  #M Calculate Geodetic height (m)
  height = np.sign(t)  * np.sqrt((x0 - xe) ** 2 + (y0 - ye) ** 2)

  #M Calculate geocentric latitude (radians).
  lat0r = np.arctan2(ye, xe)

  #M Calculate Geodetic lat and long
  geod_lat_r = np.arctan(np.tan(lat0r) * a2 / b2)
  geod_long_r = np.arctan2(y, x)		
  geod_lat = np.degrees(geod_lat_r) 	        #M Lat in degrees
  geod_long = np.degrees(geod_long_r)		#M Long in degrees

  return geod_lat, geod_long, height
