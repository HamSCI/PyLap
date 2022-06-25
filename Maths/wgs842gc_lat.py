#M
#M Name:
#M   wgs842gc_lat.m
#M
#M Purpose:
#M   Convert WGS84 ellipsoidal geodetic latitude to geocentric latitude at a 
#M   given geodetic height above the ground. 
#M      NB 1. Geocentric longitude and geodetic longitude have the same value. 
#M         2. Geodetic latitude is defined wrt the normal to the ellipsoid and
#M            is constant with height above the ellipsoid.
#M         3. Geocentric latitude is defined wrt the centre of the Earth. Thus it 
#M            NOT cons.tant with height above the ellipsoid (only true for a 
#M            spheroid)
#M 
#M Calling sequence:
#M   gc_lat = wgs842gc_lat(geod_lat, height)
#M
#M Inputs:
#M   geod_lat  -  geodetic longitude (degrees)
#M   height    -  geodetic height above the ground (m)
#M
#M Outputs:
#M   gc_lat  -  geocentric latitude (degrees)
#M 
#M Dependencies:
#M   none
#M
#M Modification history:
#M   08/11/2007 M. A. Cervera
#M     Initial version.
#M
#    20/06/2020 W.C. Liles
#       Convert to Python
#
#
import numpy as np
#
def wgs842gc_lat(geod_lat, height):

    # rtod = np.degrees(1)
    # dtor = np.radians(1)
    a = 6378137.0              #M WGS84 semimajor axis (m)
    f = 1.0 / 298.257223563   #M WGS84 flattening factor
    b = a * (1 - f)	                #M WGS84 semiminor axis (m)
    e2 = (a ** 2 - b ** 2) / a ** 2    #M eccentricity of ellipsoid squared
     
    #M Convert input WGS84 geodetic lat, long to radians
    geod_lat_r = np.radians(geod_lat) 

    #M Calculate the geocentric latitude
    chi = np.sqrt(1 - e2 * np.sin(geod_lat_r) ** 2)
    c1 = a + chi * height
    tan_gc_lat = np.tan(geod_lat_r) * (c1 - a * e2) / c1
    gc_lat = np.degrees(np.arctan(tan_gc_lat))
    return gc_lat
