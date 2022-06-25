#!/usr/bin/env python3
#M 
#M Name :
#M   coning.m
#M 
#M Purpose :
#M   Calculates correction to ray azimuth due to cone effect of linear
#M   arrays. If the radar antenna beam direction is given by "radar_bearing" and
#M   antenna array bore direction is "radar_bore", then the "ray_bearing will" be
#M   given by 
#M
#M       ray_bearing = radar_bearing + coning_correction 
#M
#M   where coning_correction is the correction due to the cone effect and given by
#M
#M       coning_correction = coning(elev, radar_bearing - radar_bore)
#M    
#M   and "elev" is the ray elevation.
#M
#M Calling sequence :
#M    coning_correction = coning(elev, off_bore)
#M
#M Inputs :
#M   elev - the ray elevation (degrees, scaler)
#M   off  - the azimuth of the radar bearing from radar bore (degrees, scaler)
#M
#M Outputs :
#M   coning_correction - the correction to the ray azimuth (degrees)
#M 
#M Modification History :
#M 26/10/2005  V1.0  M. A. Cervera  Author.
#M
#M 07/04/2009  V1.1  M. A. Cervera
#M   Added error checking to make sure that the input elevation does not
#M   exceed 90 - off_bore angle.
#M
#
#  27/06/2020 W. C. Liles
#     convert to Python
#
#
import sys
import numpy as np
#
#
#function coning_correction = coning(elev, off_bore)
def coning(elev, off_bore):
    # deg2rad = pi ./ 180   #M degrees to radians conversion
    # rad2deg = 180 ./ pi   #M radians to degrees conversion
    
    if elev > 90-off_bore: 
        print('input elevation is > 90 - off_bore angle')
        sys.exit()
    sin_elev = np.sin(np.radians(elev))
    sin_off = np.sin(np.radians(off_bore))
    temp = np.sqrt(np.abs(1.e0 - sin_off **2 - sin_elev ** 2))
    coned = np.degrees(90 - np.arctan2(temp, sin_off))
    coning_correction = coned - off_bore
    return coning_correction
