#!/usr/bin/env python3

import numpy as np  # py
import time
import ctypes as c
from pylap.raytrace_2d import raytrace_2d 
# from Ionosphere import gen_iono_grid_2d as gen_iono
from Ionosphere import gen_SAMI3_iono_grid_2d as gen_iono
from Plotting import plot_ray_iono_slice as plot_iono
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
plt.switch_backend('tkagg')

# Constants

R12 = 100                    #M R12 index
speed_of_light = 2.99792458e8
ray_bear = 78.54576739139021              #M bearing of rays
origin_lat = 40.6683           #M latitude of the start point of ray
origin_long = -105.0384          #M longitude of the start point of ray
irregs_flag = 0              #M no irregularities - not interested in
                             #M Doppler spread or field aligned irregularities

# iopnospheric params
range_step = 10
height_step = 5
Distance = 3000
datetime = dt.datetime(2023, 10, 13, 16,20)
filepath = '/home/devindiehl2/PyLap-Dev/PyLap/Examples/sami3_eclipse_2023.nc'

# make ionosphere object
print('Generating ionospheric grid... ')
ionosphere = gen_iono.gen_SAMI3_iono_grid_2d(filepath,
                                             ray_bear,
                                             origin_lat,
                                             origin_long,
                                             range_step, 
                                             height_step,
                                             Distance, 
                                             datetime)

# generate 2D slice of ionosphere
electron_density = np.transpose(ionosphere.get_2d_profile())
iono_en_grid = electron_density

#set all other inputs to zero
iono_en_grid_5 = np.zeros(electron_density.shape)
collision_freq = np.zeros(electron_density.shape)
irreg = np.zeros((4, 300))



#M convert plasma frequency grid to  electron density in electrons/cm^3
# iono_en_grid = (iono_pf_grid ** 2) / 80.6164e-6
# iono_en_grid_5 = (iono_pf_grid_5 ** 2) / 80.6164e-6

# ray tracing parameters
elevs = np.arange(1, 100, 2, dtype = float) # py
num_elevs = len(elevs)
freq = 14.0                  #M ray frequency (MHz)
freqs = freq * np.ones(num_elevs, dtype = float) # py
tol = [1e-7, 0.01, 10] 
nhops = 1                    #M number of hops to raytrace
start_height =90
# height_inc = 5

print('Generating {} 2D NRT rays ...'.format(num_elevs))

ray_data, ray_path_data, ray_path_state = \
   raytrace_2d(origin_lat, origin_long, elevs, ray_bear, freqs, nhops,
               tol, irregs_flag, iono_en_grid, iono_en_grid_5,
 	       collision_freq, start_height, height_step, range_step, irreg)




# here a Subset of the data is specified to be plotted 
start_range = 0
start_range_idx = int(start_range/range_step) 
end_range = 2000
end_range_idx = int((end_range) / range_step) + 1
start_ht = start_height
start_ht_idx = 0
end_ht = 400
end_ht_idx = int((end_ht - start_height) / height_step) + 1
iono_pf_subgrid = iono_en_grid[start_ht_idx:end_ht_idx,start_range_idx:end_range_idx]


ax, ray_handle = plot_iono.plot_ray_iono_slice(iono_pf_subgrid, start_range,
                      end_range, range_step, start_ht, end_ht, height_step,
                      ray_path_data,linewidth=1.5, color=[1, 1, 0.99])


fig_str_a = '  {}MHz   R12 = {}'.format( freq, R12)
fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
             origin_lat, origin_long, ray_bear)

fig_str = fig_str_a + fig_str_b

ax.set_title(fig_str)




plt.show()