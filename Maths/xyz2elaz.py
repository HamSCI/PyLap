#M
#M Name:
#M   xyz2elaz.m
#M
#M Purpose:
#M   Calculate the elevation and azimuth of a point with respect to an origin 
#M   on the Earth. The point is defined in cartesian coordinates (x, y, z) 
#M   relative to the specified origin where the x axis of the cartesian 
#M   coordinate frame passes through the equator at the prime meridian, y through
#M   the equator at longitude of +90 degrees, and z through the geographic north 
#M   pole. The location of the origin is specified by its geodetic (WGS84) 
#M   latitude and longitude. 
#M 
#M Calling sequence:
#M   [elev, azim] = xyz2elaz(dir_x, dir_y, dir_z, lat, lon)
#M
#M Inputs (all double precision):
#M   dir_x   - x,y,z coordinates (m) of the point relative to the specified
#M   dir_y     local origin on the Earth.
#M   dir_z  
#M   lat     - Geodetic (WGS84) latitude and longitude (degrees) of location from
#M   lon       which point is defined.
#M
#M Outputs (all double precision):
#M   elev      - elevation of point from local origin (degrees)
#M   azim      - azimuth of point form local origin (degrees)
#M 
#M Dependencies:
#M   None.
#M
#M Modification history:
#M   26/11/2009 M. A. Cervera
#M     Initial version: xyz2relaz.f90
#M
#M   5/2/2014 D. J. Netherway
#M     Converted to matlab
#M     Also removed "abs" from elev_r calculation to allow negative elevations.
#M     Applies to both spherical geometry and WGS84 because elevation is
#M     referenced to the local plane orthogonal to the up defined by the
#M     lattitude. But careful not to mix geodetic and geocentric environments
#M     because the up direction differs between the two systems.
#M
#   6/8/2020 W. C. Liles
#       Convert to Python
#
#
import numpy as np
#
#
# function [elev, azim] = xyz2elaz(dir_x, dir_y, dir_z, lat, lon)
def xyz2elaz(dir_x, dir_y, dir_z, lat, lon):

  #M Define constants
  #M pi = 3.1415926535897931d0
  # rad2deg = 180.0d0/pi use numpy.degrees
  # deg2rad = pi/180.0   use numpy/radians

  #M determine the sin and cosine of lat and lon required for the series of 
  #M rotations to convert local cartesian frame to ENU frame
  lat_r = np.radians(lat)
  lon_r = np.radians(lon)
  sin_phi = np.sin(lat_r)
  cos_phi = np.cos(lat_r)
  sin_theta = np.sin(lon_r)
  cos_theta = np.cos(lon_r)

  #M Calculate the coodinates of the point in ENU cartesian coordinates local 
  #M to the input origin
  E =  -dir_x * sin_theta + dir_y * cos_theta
  N =  -dir_x * sin_phi * cos_theta -  \
	dir_y * sin_phi * sin_theta +  \
	dir_z * cos_phi
  U =   dir_x * cos_phi * cos_theta +  \
	dir_y * cos_phi * sin_theta +  \
	dir_z * sin_phi

  #M calculate the slant range, elevation and the azimuth of the point 
  #M relative to the input origin
  slant_rng = np.sqrt(dir_x ** 2 + dir_y ** 2 + dir_z ** 2)
  elev_r = np.arcsin(U / slant_rng)
  azim_r = np.arctan2(E, N)

  #M Convert output angles to degrees
  azim = np.degrees(azim_r)
  elev = np.degrees(elev_r)

  return elev, azim
