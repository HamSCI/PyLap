#!/usr/bin/env python3
#M Name: 
#M   latlon2raz.m
#M 
#M Purpose:
#M   Converts latitude and longitude to polar coordinates in range/azimuth.
#M
#M Calling sequence:
#M   1.  [range, azimuth] = latlon2raz(point_lat, point_lng, origin_lat, ...
#M  	      origin_lng)
#M   2.  [range, azimuth] = latlon2raz(point_lat, point_lng, origin_lat, ...
#M  	      origin_lng, geoid)
#M   3.  [range, azimuth] = latlon2raz(point_lat, point_lng, origin_lat, ...
#M  	      origin_lng, 'wgs84')
#M   4.  [range, azimuth] = latlon2raz(point_lat, point_lng, origin_lat, ...
#M  	      origin_lng, 'spherical')
#M   5.  [range, azimuth] = latlon2raz(point_lat, point_lng, origin_lat, ...
#M  	      origin_lng, 'spherical', re)
#M     
#M Inputs:
#M   point_lat  - latitude of point (array)
#M   point_lng  - longitude of point (array)
#M    
#M   origin_lat - the latitude and longitude of the origin for
#M   origin_lng      which the range, azimuth data should refer (scalar).
#M  
#M Optional Inputs
#M   geoid      - The type of ellipsoid to use. If not specfied then a 
#M                spherical Earth is assumed. The following values are valid:
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
#M   range   - range as distance from (origin_lat, origin_lng) in meters
#M   azimuth - azimuth as true bearings (degrees east of north) from origin
#M	
#M Notes: 
#M   1. The range will never exceed half the circumference of the earth.
#M      instead the returned azimuth will be shifted by 180 degrees.
#M   2. Uses T. Vincenty's iterative method for the non-spherical geoids. This 
#M      should be good over distances from a few cm to nearly 20,000 km with
#M      millimeter accuracy.
#M   3. For non-spherical geoids the solution will fail to converge near the
#M      antipodal point (within 0.01 deg in latitude or 0.1 deg in
#M      longitude). For this case the precision will be reduced.
#M 
#M Modification History:
#M 20/07/2006 M. A. Cervera 
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
#M 22/03/2013 M. A. Cervera
#M   Now returns a warning if near the antipodal point for non-spherical geoids.
#M
#  04/08/2020 W. C. Liles
#    Convert to Python
#
#
import numpy as np
#
#
#function  [range, azimuth] = latlon2raz(point_lat, point_lng, origin_lat, ...
#     origin_lng, vargin, vargin2)
def latlon2raz(point_lat, point_lng, origin_lat, origin_lng,
               geoid = 'spherical', re = 6378137): # mean radius of Earth in m 
#M
#M general constants
#M
#    dtor = pi/180 use np.radians instead
#    radeg= 180/pi use np.degrees instead
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
    #M ellipsoid constants 
    #M
    if geoid_low == 'spherical':
        #M spherical Earth
        bore_lat = origin_lat    
        bore_long = 180.0    
    
        lat = point_lat
        long = point_lng + 180.0 - origin_lng   #M as we want longtitude line between
    					   #M origin and bore to be zero
        chi      = np.radians(90.0 - bore_lat)
        rlat     = np.radians(lat)
        rlon     = np.radians(long)
        coschi   = np.cos(chi)
        sinchi   = np.sin(chi)
        coslat   = np.cos(rlat)
        coslon   = np.cos(rlon)
        sinlat   = np.sin(rlat)
        sinlon   = np.sin(rlon)
    
        y        = sinlon * coslat
        x        = coslon * coslat * coschi + sinchi * sinlat
    
        geo_lat  = np.degrees(np.arcsin(coschi * sinlat -
                                      coslon * coslat * sinchi))
        geo_long = np.degrees(np.arctan2(y, x)) + bore_long
    
        #M due south is 0 degrees long., +ve anticlockwise,
        #M due north is 0 degrees long., +ve clockwise, = true bearings
        azimuth = 180.0 - geo_long           
        azimuth = np.mod((azimuth + 360.), 360)   #M force azimuth to 0 - 360 degrees
    
        range_arg = np.abs(np.radians(90.0 - geo_lat) * re)
    else:    
        # here if ellisoid
      
        #M---------  Initial values  --------------------
        a = geoid_dict[geoid_low][0]
        f = geoid_dict[geoid_low][1]
        
        #M ellipsoid
        #M---------  Initial values  --------------------
        epsi = 0.5e-13			#M Tolerence.
        glat1 = np.radians(origin_lat)
        glon1 = np.radians( origin_lng)
        glat2 = np.radians(point_lat)
        glon2 = np.radians(point_lng)
    
        r = 1 - f
        tu1 = r * np.sin(glat1) / np.cos(glat1)
        tu2 = r * np.sin(glat2) / np.cos(glat2)
        cu1 = 1 / np.sqrt(1 + tu1 ** 2)
        su1 = cu1 * tu1
        cu2 = 1 / np.sqrt(1 + tu2 ** 2)
        s = cu1 * cu2
        baz = s * tu2
        faz = baz * tu1
        x = glon2 - glon1
    
        #M-----  Iterate  ---------------
        d = 1e10
        count = 0
        while np.max(np.abs(d-x)) > epsi and count < 20:
          count = count+1
          sx = np.sin(x)
          cx = np.cos(x)
          tu1 = cu2  * sx
          tu2 = baz - su1 * cu2 * cx
          sy = np.sqrt(tu1 ** 2 + tu2 ** 2)
          cy = s * cx + faz
          y = np.arctan2(sy, cy)
          np.seterr(invalid='ignore')
          sa = (s * sx )/ sy
          c2a = 1 - sa ** 2
          cz = 2 * faz
          w = np.where(c2a > 0)[0]
          cnt = len(w)
          if cnt > 0: 
              cz[w] = cy[w] - cz[w] / c2a[w]
          e = 2 * cz ** 2 - 1
          c = ((-3 * c2a + 4) * f + 4) * c2a * f/16
          d = x
          x = ((e * cy * c + cz) * sy * c + y) * sa
          x = (1 - c) * x * f + glon2 - glon1

        if count == 20:
          print('Near the antipodal point - solution failed to converge and ' 
    	       'has reduced precision')
          
        
        #M------  Finish up  ----------------
        faz = np.arctan2(tu1, tu2)
        baz = np.arctan2(cu1 * sx, baz * cx - su1 * cu2) + np.pi
        x = np.sqrt((1  / r ** 2 - 1) * c2a + 1) + 1
        x = (x - 2)  / x
        c = 1 - x
        c = (x ** 2  / 4 + 1) / c
        d = (0.375 * x ** 2 - 1) * x
        x = e  * cy
        s = 1 - e ** 2
        range_arg = ((((4 * sy ** 2 - 3) * s * cz * d/6 - x) * d/4 + cz)
                     * sy * d + y) * c * a * r

    
        azimuth = np.degrees(faz)
        w = np.where(azimuth < 0)[0]
        cnt = len(w)
        if cnt > 0:
            azimuth[w] = azimuth[w] + 360
     
    return range_arg, azimuth
