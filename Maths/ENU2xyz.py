#!/usr/bin/env python3
#M
#M Name:
#M   ENU2xyz.m
#M
#M Purpose:
#M   Convert the location of a point specified in an East, North, Up (ENU) frame
#M   at a local origin on the Earth to cartesian coordinates (x, y, z), where
#M   the x axis of the cartesian coordinate frame passes through the equator
#M   at the prime meridian, y through the equator at longitude of +90 degrees,
#M   and z through the geographic north pole. The ENU frame origin on the Earth 
#M   is specified by its geodetic (WGS84) latitude and longitude. 
#M
#M Calling sequence:
#M   [x, y, z] = ENU2xyz(E, N, U, lat, lon)
#M
#M Inputs: 
#M   E, N, U   - East, North, Up coordinates of point of interest (m),  may be
#M               arrays of any (but identical) shape 
#M   lat, lon  - Geodetic (WGS84) latitude and longitude (degrees) of location  
#M               of the (E,N,U) frame origin. If E,N,U are arrays then lat, lon
#M               must either be be scalar or arrays of identical shape to E,N,U.
#M               If E,N,U are scalar then lat, lon may be arrays of any (but
#M               identical) shape 
#M
#M Outputs:
#M   x, y, z  - cartesian coordinates of the point (m) relative to the ENU
#M              frame origin
#M 
#M Dependencies:
#M   None.
#M
#M Modification history:
#M   27/11/2009  V1.0  M. A. Cervera
#M     Initial version.
#M
#M   14/11/2012  V1.1  M. A. Cervera
#M     Inputs can now be arrays.
#M
#    01/07/2020  W. C. Liles
#       Convert to Python
#
import numpy as np
#
#function [x, y, z] = ENU2xyz(E, N, U, lat, lon)
def  ENU2xyz(E, N, U, lat, lon):
  #M Define constants
  #deg2rad = pi/180.0         #M radians to degrees conversion
     
  #M determine the sin and cosine of lat and lon required for the series of 
  #M rotations to convert local ENU frame to cartesian frame
  lat_r = np.radians(lat) 
  lon_r = np.radians(lon)
  sin_phi = np.sin(lat_r)
  cos_phi = np.cos(lat_r)
  sin_theta = np.sin(lon_r)
  cos_theta = np.cos(lon_r)

  #M Perform rotations to tranform local ENU coordinates of the point to
  #M x,y,z cartesian coordinates
  x = -E * sin_theta - N * sin_phi * cos_theta + U * cos_phi * cos_theta
  y =  E * cos_theta - N * sin_phi * sin_theta + U * cos_phi * sin_theta
  z =  N * cos_phi   + U * sin_phi

  return x, y, z

