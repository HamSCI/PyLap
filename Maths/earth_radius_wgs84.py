#M
#M Name:
#M   earth_radius_wgs84.m
#M
#M Purpose:
#M   Returns the "radius" of the Earth (i.e. distance from centre of the Earth
#M   to the surface) at a given input WGS84 geodetic latitude.
#M	
#M Calling sequence:
#M     re_wgs84 = earth_radius_wgs84(geod_lat)
#M
#M Inputs:	
#M   geod_lat - WGS84 geodetic latitude (degrees)
#M
#M Outputs:
#M   re_wgs84 - "radius" of the Earth at the input geodetic latitude (m)
#M
#M Dependencies:
#M   none
#M
#M Modification History:
#M 08/11/2007 M. A. Cervera
#M   Initial version. 
#M
#     21/06/2020 W. C. Liles
#    Python implementation
#
import numpy as np
#
def earth_radius_wgs84(geod_lat):
    sem_maj_ax = 6378137.0         #M semi-major axis of WGS84 (m)
    sem_min_ax = 6356752.314245    #M semi-minor axis of WGS84 (m)
#    dtor = pi / 180.0

    a2 = sem_maj_ax ** 2
    b2 = sem_min_ax ** 2
    a2cos2l  = a2 * np.cos(np.radians(geod_lat)) ** 2
    b2sin2l  = b2 * np.sin(np.radians(geod_lat)) ** 2

    re_wgs84 = np.sqrt((a2 * a2cos2l + b2 * b2sin2l) / (a2cos2l + b2sin2l))
    return re_wgs84
