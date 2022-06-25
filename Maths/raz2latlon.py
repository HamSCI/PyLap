#M
#M Name:
#M   raz2latlon.m
#M
#M Purpose:
#M   Converts the ground range and azimuth (from true North) of point from a 
#M   particular origin on the Earth to latitude and longitude.
#M	
#M Calling sequence:
#M   1.  [geo_lat, geo_long] = raz2latlon(range, azim, origin_lat, origin_long)
#M   2.  [geo_lat, geo_long] = raz2latlon(range, azim, origin_lat, ...
#M                                        origin_long, geoid)
#M   3.  [geo_lat, geo_long] = raz2latlon(range, azim, origin_lat, ...
#M                                        origin_long, 'wgs84')
#M   4.  [geo_lat, geo_long] = raz2latlon(range, azim, origin_lat, ...
#M                                        origin_long, 'spherical')
#M   5.  [geo_lat, geo_long] = raz2latlon(range, azim, origin_lat, ...
#M                                        origin_long, 'spherical', re)
#M
#M
#M Inputs:	
#M   range 	Array	range (m) of point from the origin.
#M   azim 	Array	azimuth of point from true North with respect to the 
#M                       origin (degrees).
#M   origin_long	Scalar	longitude of origin (degrees).	
#M   origin_lat	Scalar	latitude of origin (degrees).
#M
#M Optional Inputs
#M   geoid       String  The type of ellipsoid to use. If not specfied then a 
#M                       spherical Earth is assumed. The following values are
#M                       valid:
#M
#M     'spherical'   - Spherical Earth with radius = 6378137 m (default) or 
#M                     user specified input (see calling sequence example 5.)
#M     'wgs84'       - WGS84 
#M     'wgs1972'     - WGS 1972
#M     'grs80'       - Geodetic Reference System 1980 (GRS 80)
#M     'airy'        - Airy
#M     'aust_nat'    - Australian National
#M     'bessel_1841' - Bessel 1841 
#M     'clarke_1866' - Clarke 1866
#M     'clarke_1880' - Clarke 1880
#M   
#M Outputs:
#M   geo_lat	Array	latitude of point (degrees).
#M   geo_long	Array	longitude of point (degrees).
#M
#M Notes:
#M   1. Uses T. Vincenty's method for the non-spherical geoids. This should be
#M      good over distances from a few cm to nearly 20,000 km with millimeter 
#M      accuracy.
#M
#M Modification History:
#M 09/02/1996 M. A. Cervera 
#M   Initial version - spherical earth only
#M
#M 26/06/2007 M. A. Cervera
#M   Added various ellipsoids: WGS84, WGS 1972, GRS 80, Airy, Australian
#M   National, Bessel 1841, Clarke 1866, Clarke 1880. Uses T. Vincenty's
#M   method. Converted to matlab from FORTRAN code at 
#M   ftp://www.ngs.noaa.gov/pub/pcsoft/for_inv.3d/source/forward.for
#M
#M 25/10/2012 M. A. Cervera
#M   User can now specify the radius of the Earth for the spherical case
#M
#M 17/12/2013 L. H. Pederick
#M   Minor modification for efficiency - added 'var' argument to exist()
#M   function calls
#
#  06/07/2020 W. C. Liles
#    Convert to Python
#
import numpy as np
#
# function [geo_lat, geo_long] = raz2latlon(range, azim, origin_lat, ...
#                                          origin_long, vargin, vargin2)
#  have geoid and earth radius (er) optional arguments
def  raz2latlon(range_arg, azim, origin_lat, origin_long, geoid = 'spherical',
                re = 6378137):        # mean radius of Earth in m 
    #M
    #M general constants
    #M
    
    #M
    #M ellipsoid constants 
    #M
    # for each geoid define two numbers in list 
    #    [a = semi-major axis (m),   f = Flattering factor]
    geoid_dict = {
       'wgs84' : [6378137.0, 1.0 / 298.257223563], # WGS84 (www.wgs84.com)
       'airy'  : [6377563.396, 1.0 / 299.3249646], # Airy
       'aust_nat' : [6378160.0, 1.0 / 298.25], # Australian National
       'bessel_1841' : [6377397.155, 1.0 / 299.1528128], # Bessel 1841
       'clarke_1866' : [6377397.155, 1.0 / 299.1528128], # Clark 1886
       'clarke_1880' : [6378249.145, 1.0 / 293.465000], # Clarke 1880
       'grs80' : [6378137.0, 1.0 / 298.257222101], # Geodetic Reference System 1980 (GRS 80)
       'wgs1972':[6378135.0, 1.0 / 298.26,], #M WGS 1972
       'spherical' : [0, 0]} # spherical earth
    geoid_low = geoid.lower()

    #M
    #M do the calculations
    #M
    if geoid_low == 'spherical':
         #M spherical Earth
         latbore = np.pi/2 - range_arg/re      #Mlat. of point in bore system
         longbore = np.radians(180 - azim)         #Mlong. of point in bore system
    
         chi        = np.radians(90.0 - origin_lat)
         coschi     = np.cos(chi)
         sinchi     = np.sin(chi)
         coslatb    = np.cos(latbore)
         coslonb    = np.cos(longbore)
    
         sinlatb    = np.sin(latbore)
         sinlonb    = np.sin(longbore)
    
         geo_lat    = np.degrees(np.arcsin(coschi*sinlatb -
                                           coslonb*coslatb*sinchi))
         y          = sinlonb * coslatb
         x          = coslonb * coslatb * coschi + sinchi * sinlatb
    
         geo_long   = np.degrees(np.arctan2(y,x)) + origin_long
      
      
    else:    
        # here if ellisoid
      
        #M---------  Initial values  --------------------
        a = geoid_dict[geoid_low][0]
        f = geoid_dict[geoid_low][1]
        epsi = 0.5e-13			#M Tolerence.
        glon1 = np.radians(origin_long)
        glat1 = np.radians(origin_lat)
        faz = np.radians(azim)
    
        r = 1 - f
        # tu = repmat(r * sin(glat1) / cos(glat1), length(faz))
        tu = np.tile(r * np.sin(glat1) / np.cos(glat1),
                     len(faz))
        sf = np.sin(faz)
        cf = np.cos(faz)
        baz = 0 * faz		#M Want as many baz az faz.
        cnt = np.count_nonzero(cf)
        w = np.where( cf != 0)
        if (cnt > 0): 
          baz[w] = np.arctan2(tu[w], cf[w]) * 2
        
    
        cu = 1 / np.sqrt(tu ** 2 + 1)
        su = tu * cu
        sa = cu * sf
        c2a = 1 - sa ** 2
        x = np.sqrt((1 / r ** 2 - 1) * c2a + 1) + 1
        x = (x - 2) / x
        c = 1 - x
        c = (x ** 2 / 4 + 1) / c
        d = (0.375 * x ** 2 - 1) * x
        tu = range_arg / r / a / c
        y = tu
    
        #M------  Iterate  --------------
        while np.max(np.abs(y-c)) > epsi:
          sy = np.sin(y)
          cy = np.cos(y)
          cz = np.cos(baz + y)
          e = 2. * cz ** 2. - 1.
          c = y
          x = e * cy
          y = 2. * e - 1.
          y = ( (y * cz * d * (4. * sy ** 2. - 3.)/6. + x) * d/4. - cz) * sy *d + tu
    
        #M-------  Finish up  -----------
        baz = cu * cy * cf - su * sy
        c = r * np.sqrt(sa ** 2 + baz ** 2)
        d = su * cy + cu * sy * cf
        glat2 = np.arctan2(d, c)
        c = cu * cy - su * sy * cf
        x = np.arctan2(sy * sf, c)
        c = ( (-3. *c2a + 4.) * f + 4.) * c2a * f / 16.
        d = ( (e * cy * c + cz) * sy * c + y) * sa
        glon2 = glon1 + x - (1 - c) * d * f
        baz = np.arctan2(sa, baz) + np.pi
        geo_long = np.degrees(glon2)
        geo_lat = np.degrees(glat2)
        azi2 = np.degrees(baz)
    return geo_lat, geo_long
    
