#!/usr/bin/env python3

import numpy as np  # py
import time
import ctypes as c
from pylap.raytrace_2d import raytrace_2d 
# from Ionosphere import gen_iono_grid_2d as gen_iono
from Ionosphere import gen_SAMI3_iono_grid_2d as gen_iono
from Plotting import plot_ray_iono_slice as plot_iono
import matplotlib.pyplot as plt

plt.switch_backend('tkagg')

#M
#M setup general stuff
#M

R12 = 100                    #M R12 index
speed_of_light = 2.99792458e8
  
elevs = np.arange(2, 62, 2, dtype = float) # py
num_elevs = len(elevs)
freq = 15.0                  #M ray frequency (MHz)
freqs = freq * np.ones(num_elevs, dtype = float) # py
ray_bear = 324.7             #M bearing of rays
origin_lat = -23.5           #M latitude of the start point of ray
origin_long = 133.7          #M longitude of the start point of ray
tol = [1e-7, 0.01, 10] # py
nhops = 1                    #M number of hops to raytrace
doppler_flag = 1             #M generate ionosphere 5 minutes later so that
                             #M Doppler shift can be calculated
irregs_flag = 0              #M no irregularities - not interested in
                             #M Doppler spread or field aligned irregularities
kp = 0                       #M kp not used as irregs_flag = 0. Set it to a
                             #M dummy value


print('\n'
       'Example of 2D numerical raytracing for a fan of rays for a WGS84'
       ' ellipsoidal Earth\n\n') # py

#M
#M generate ionospheric, geomagnetic and irregularity grids
#M
max_range = 10000       #M maximum range for sampling the ionosphere (km)
num_range = 201        #M number of ranges (must be < 2000)
range_inc = max_range / (num_range - 1) # py
start_height = 0        #M start height for ionospheric grid (km)
height_inc = 3          #M height increment (km)
num_heights = 200      #M number of  heights (must be < 2000)



print('Generating ionospheric grid... ')
ionosphere = gen_iono.gen_SAMI3_iono_grid_2d('', )


#M convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid = (iono_pf_grid ** 2) / 80.6164e-6
iono_en_grid_5 = (iono_pf_grid_5 ** 2) / 80.6164e-6




#M
#M Example 1 - Fan of rays, 10 MHz, single hop. Print to encapsulated
#M postscript and PNG. Note the transition from E-low to E-High to F2-low modes.
#M

#M call raytrace for a fan of rays
#M first call to raytrace so pass in the ionospheric and geomagnetic grids
print('Generating {} 2D NRT rays ...'.format(num_elevs))

ray_data, ray_path_data, ray_path_state = \
   raytrace_2d(origin_lat, origin_long, elevs, ray_bear, freqs, nhops,
               tol, irregs_flag, iono_en_grid, iono_en_grid_5,
 	       collision_freq, start_height, height_inc, range_inc, irreg)


################
### Figure 1 ###
################
start_range = 0
start_range_idx = int(start_range/range_inc) 
end_range = 3000
end_range_idx = int((end_range) / range_inc) + 1
start_ht = start_height
start_ht_idx = 0
end_ht = 400
end_ht_idx = int(end_ht / height_inc) + 1
iono_pf_subgrid = iono_pf_grid[start_ht_idx:end_ht_idx,start_range_idx:end_range_idx]


ax, ray_handle = plot_iono.plot_ray_iono_slice(iono_pf_subgrid, start_range,
                      end_range, range_inc, start_ht, end_ht, height_inc,
                      ray_path_data,linewidth=1.5, color=[1, 1, 0.99])
# 

fig_str_a = '{}/{}/{}  {:02d}:{:02d}UT   {}MHz   R12 = {}'.format(
              UT[1], UT[2], UT[0], UT[3], UT[4], freq, R12)
fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
             origin_lat, origin_long, ray_bear)

fig_str = fig_str_a + fig_str_b

ax.set_title(fig_str)




plt.show()