#M
#M Name:
#M   relaz2xyz.m
#M
#M Purpose:
#M   Convert the location of a point specified in terms of slant-range, 
#M   elevation, and azimuth from a local origin on the Earth to cartesian 
#M   coordinates (x, y, z), where the x axis of the cartesian coordinate frame 
#M   passes through the equator at the prime meridian, y through the equator at 
#M   longitude of +90 degrees, and z through the geographic north pole. The local
#M   origin on the Earth  is specified by its geodetic (WGS84) latitude and 
#M   longitude. 
#M
#M Calling sequence:
#M   call relaz2xyz(slant_rng, elev, azim, lat, lon, point_x, point_y, point_z)
#M
#M Inputs (all real*8):
#M   slant_rng - distance of point from local origin (m)
#M   elev      - elevation of point from local origin (degrees)
#M   azim      - azimuth of point form local origin (degrees)
#M   lat       - Geodetic (WGS84) latitude and longitude (degrees) of location  
#M   lon         from which point is defined.
#M
#M Outputs (all real*8):
#M   point_x   - x, y, z cartesian coordinates of the point (m) relative to the 
#M   point_y     input origin
#M   point_z  
#M 
#M Dependencies:
#M   None.
#M
#M Usage for directions is to call with slant_rng set to 1.0
#M
#M Modification history:
#M   26/11/2009 M. A. Cervera
#M     Initial version. relaz2xyz.f90
#M
#M   6/1/2015 D. J. Netherway
#M     Converted to matlab
#
#   7/8/2020 W. C. Liles
#      Converted to Python
#     slang_rng, elev, azim can be scalar values or numpy arrays
#
import numpy as np

# function [point_x, point_y, point_z] = relaz2xyz(slant_rng, elev, azim, lat, lon)
def relaz2xyz(slant_rng, elev, azim, lat, lon):
  cosd = lambda x : np.cos(np.radians(x))
  sind = lambda x : np.sin(np.radians(x))
  #M Calculate the coodinates of the point in local ENU (East, North, Up) 
  #M cartesian coordinates
  E = slant_rng * sind(azim) * cosd(elev)    #M East
  N = slant_rng * cosd(azim) * cosd(elev)    #M North
  U = slant_rng * sind(elev)                 #M Up

  #M determine the sin and cosine of lat and lon required for the series of 
  #M rotations to convert local cartesian frame to ENU frame
  sin_phi = sind(lat)
  cos_phi = cosd(lat)
  sin_theta = sind(lon)
  cos_theta = cosd(lon)

  #M Perform rotations to tranform local ENU coordinates of the point to local
  #M x,y,z cartesian coordinates
  point_x = -E * sin_theta - N * sin_phi * cos_theta + U * cos_phi * cos_theta
  point_y =  E * cos_theta - N * sin_phi * sin_theta + U * cos_phi * sin_theta
  point_z =  N * cos_phi   + U * sin_phi

  return point_x, point_y, point_z
  

