#!/usr/bin/env python3
import math
import numpy as np
import scipy.interpolate as spi
import netCDF4 as nc
# from Sami3Entitties import SAMI3Params
from Maths import wgs84_xyz2llh
import matplotlib as plt
import pyproj as pj
import sys
from Maths import earth_radius_wgs84
from Maths import wgs842gc_lat
from Maths import raz2latlon
from Maths import eff_coll_freq



# note this should be an object once this is past prototyping 
def gen_iono_grid_SAMI3_2d(origin_lat, origin_lon, R12, UT, azim, 
            max_range, num_range, range_inc, start_height,
            height_inc, num_heights, kp, doppler_flag,
            filePath, *args):
    
    # print(gen_iono_grid_SAMI3_2d)

    sami_data = GetSAMIDataFromPath(filePath)
     # XY we need way to convert pylap user given specifications to sami data
    #    we need to fix a lot of the data structure conversion to make this section work
  
    for dim in sami_data.dimensions.values():
        print (dim)
    for var in sami_data.variables.values():
        print (var)


    # fixed time works for us for now we can figure out conversion later
    fixed_time = 50 # time step : 0 to 479
    latc = np.array(sami_data.variables['lat0G'])
    lonc = np.array(sami_data.variables['lon0G'])
    altc = np.array(sami_data.variables['alt0G'])
    denc = np.array(sami_data['dene0G'][fixed_time,:,:,:])
    
    lat,lon = np.meshgrid(latc,lonc)

    samples = 800
    del_distance = 5000
    bearing =80
    wwv_10_lon= -105.0403
    wwv_10_lat = 40.6799

    earth_elipse = pj.Geod(ellps='WGS84')
    path = earth_elipse.fwd_intermediate(wwv_10_lon, wwv_10_lat, bearing, samples, del_distance)
    path_range=[del_distance*x for x in range(samples)]



    lons = np.array(path.lons)
    lats = np.array(path.lats)

    lons = convert_lon_array_to_360(lons)

    points = list(zip(lon.ravel(), lat.ravel()))
    path_pts = list(zip(lons,lats))

    # for each altiude, we need to get the density values along the path

    # NOTE I will store this in an array that has 'samples' length, vstacked on top of each other

    density_slice = []

    for idx, alt in enumerate(altc): # this will need a new interpolating function for n samples with 100 runs, be prepared

    

        #r = alt+6371  # not needed for linear interpolation

        den_grid = denc[:,idx,:]

        values = den_grid.ravel() # NOTE the point positions won't change, so there is no reason to re-call

                                  # but, it does still leave us with the elliptical vs linear position

        interp = spi.LinearNDInterpolator(points, values)

    

        den_strip = interp(path_pts) # each of these is a length 'samples' from above

        if idx ==0:

            density_slice.append(np.array(den_strip)) # this is appending lengthwise

            density_slice = np.array(density_slice)

        else:

            # the first loop I ran used the list append features all the way through, this seems to be on par time-wise

            # more testing is probably necessary, but it's not what I'm trying to optimize right this second

            density_slice = np.vstack((density_slice, np.array(den_strip)))
            print(density_slice)
    return density_slice        
    # xx,yy =np.meshgrid(path_range,altc)


def GetSAMIDataFromPath(FilePath):
    print(FilePath)
    sami_data = nc.Dataset(FilePath)
    return sami_data


def convert_lon_array_to_360(lon_array):

    '''convert longitude (-180 to 180) array to spherical coordinates (0 to 360)

    this is essential for interpolation in the same spatial region defined by the interpolate'''

    for idx,elm in enumerate(lon_array):

        if lon_array[idx] < 0:

            # in sami3 data 0 is the prime meridian so -179 corresponds to 181

            lon_array[idx] = elm + 360

 

    # careful to keep this return separate

    return lon_array