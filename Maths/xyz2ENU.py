#M
#M Name:
#M  xyz2ENU.m
#M
#M Purpose:
#M   Convert the location of a point specified in a cartesian coordinate (x,y,z) 
#M   frame at a local origin on the Earth to East, North, Up (ENU), where
#M   the x axis of the cartesian coordinate frame passes through the equator
#M   at the prime meridian, y through the equator at longitude of +90 degrees,
#M   and z through the geographic north pole. The (x,y,z) frame origin on the
#M   Earth is specified by its geodetic (WGS84) latitude and longitude. 
#M
#M Calling sequence:
#M   [E, N, U] = xyz2ENU(x, y, z, lat, lon)
#M
#M Inputs: 
#M   x, y, z   - cartesian coordinates of the point (m) relative to frame origin,
#M               may be arrays of any (but identical) shape
#M   lat, lon  - Geodetic (WGS84) latitude and longitude (degrees) of location  
#M               of the (x,y,z) frame origin. If x,y,z are arrays then lat, lon
#M               must either be be scalar or arrays of identical shape to x,y,z.
#M               If x,y,z are scalar then lat, lon may be arrays of any (but
#M               identical) shape 
#M
#M Outputs:
#M   E, N, U   - East, North, Up coordinates of point of interest (m) relative
#M               to the (x,y,z) frame origin
#M 
#M Dependencies:
#M   None.
#M
#M Modification history:
#M   12/10/2012  V1.0  M. A. Cervera
#M     Initial version.
#M
#M   14/11/2012  V1.1  M. A. Cervera
#M     Inputs can now be arrays.
#M
#
#     01/07/2020 W. A. Liles
#       Convert to Python
#
import numpy as np

# function [E, N, U] = xyz2ENU(x, y, z, lat, lon)
def xyz2ENU(x, y, z, lat, lon):
  #M Define constants
  # deg2rad = pi/180.0         #M radians to degrees conversion
  #M determine the sin and cosine of lat and lon required for the series of 
  #M rotations to convert local cartesian frame to ENU frame
  lat_r = np.radians(lat)
  lon_r = np.radians(lon) 
  sin_phi = np.sin(lat_r)
  cos_phi = np.cos(lat_r)
  sin_theta = np.sin(lon_r)
  cos_theta = np.cos(lon_r)

#M   #M define the rotation matricies 
#M   Rot1 = [1  0        0         
#M           0  sin_phi  cos_phi   
#M 	    0 -cos_phi  sin_phi] 
#M 	
#M   Rot2 = [-sin_theta  cos_theta 0   
#M           -cos_theta -sin_theta 0  
#M 	     0          0         1] 
#M 
#M   #M form the vector of the input coordinates for the rotation matricies to 
#M   #M operate upon - make sure we record the shape of the original input
#M   size_x = size(x)
#M   xyz = [x(:) y(:) z(:)]'
#M   
#M   #M Perform rotations to transform local x,y,z cartesian coordinates of the
#M   #M point to ENU coordinates  
#M   ENU = Rot1 * (Rot2 * xyz)
#M   
#M   #M reshape the result to have the same shape as the input 
#M   E = squeeze(ENU(1, :))
#M   N = squeeze(ENU(2, :))
#M   U = squeeze(ENU(3, :))
#M 
#M   E = reshape(E, size_x)
#M   N = reshape(N, size_x)
#M   U = reshape(U, size_x)
  
  E = -x * sin_theta          + y * cos_theta
  N = -x * cos_theta * sin_phi - y * sin_theta * sin_phi + z * cos_phi
  U =  x * cos_theta * cos_phi + y * sin_theta * cos_phi + z * sin_phi
  
  
  return  E, N, U
