
from cgi import print_environ
import numpy as np  # py
import time
import ctypes as c
import os 

# os.system('python3 setup.py install --user')
from pylap.raytrace_2d import raytrace_2d 
from Ionosphere import gen_iono_grid_2d as gen_iono
from Plotting import plot_ray_iono_slice as plot_iono

#import raytrace_2d as raytrace
# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.switch_backend('tkagg')
#M
#M setup general stuff
#M
UT = [2001, 3, 15, 7, 0]#2001,3,15,14,15 makes 7:00UT    #M UT - year, month, day, hour, minute
#  The above is not a standare Python datetime object but leaving unchanged
R12 = 100                    #M R12 index
speed_of_light = 2.99792458e8
  
#elevs = [2:2:60]             #M initial ray elevation
elevs = np.arange(2, 62, 2, dtype = float) # py
# num_elevs = length(elevs)
num_elevs = len(elevs)


freq = 10.0  # M ray frequency (MHz)
#X based on error, converting to kilometers:
origin_ht = 1.599 # km    =5200 altitude + 47 ft tower (meaningless precision at this scale)

# freqs = freq.*ones(size(elevs))
freqs = freq * np.ones(num_elevs, dtype = float) # py
ray_bear =85            #M bearing of rays

#X WWV 10 MHz antenna location, I suspect this is a little off
#origin_lat = convert_latlon(40,40,47.8)
origin_lat  = 40.6799
origin_long = -105.0403

#tol = [1e-7 .01 10]          #M ODE tolerance and min/max step sizes
tol = [1e-7, 0.01, 10] # py
nhops = 1                    #M number of hops to raytrace
doppler_flag = 1             #M generate ionosphere 5 minutes later so that
                             #M Doppler shift can be calculated
irregs_flag = 0              #M no irregularities - not interested in
                             #M Doppler spread or field aligned irregularities
kp = 0                       #M kp not used as irregs_flag = 0. Set it to a
                             #M dummy value

#fprintf( ['\n' ...
#  'Example of 2D numerical raytracing for a fan of rays for a WGS84 ellipsoidal' ...
#  ' Earth\n\n'])
print('\n'
       'Example of 2D numerical raytracing for a fan of rays for a WGS84'
       ' ellipsoidal Earth using the Sami3 model\n\n') # py

#M
#M generate ionospheric, geomagnetic and irregularity grids
#M
max_range = 10000       #M maximum range for sampling the ionosphere (km)  # this might be fixed for sami3
num_range = 201         #M number of ranges (must be < 2000)
# range_inc = max_range ./ (num_range - 1)   #M range cell size (km)
range_inc = max_range / (num_range - 1) # py

start_height = 0        #M start height for ionospheric grid (km)
height_inc = 3          #M height increment (km)
num_heights =200       #M number of  heights (must be < 2000)

# clear iri_options
#iri_options.Ne_B0B1_model = 'Bil-2000'  #M this is a non-standard setting for
                                        #M IRI but is used as an example
# implement the above by means of dictionay
sami_options = {
               'data_path': '/home/xy/hamsci/sami_pylap/data/sami3models/sami3_eclipse_2023.nc',
              }   # py
print('Generating ionospheric grid... ')
iono_pf_grid, iono_pf_grid_5, collision_freq, irreg, iono_te_grid = \
    gen_iono.gen_iono_grid_2d(origin_lat, origin_long, R12, UT, ray_bear,
             max_range, num_range, range_inc, start_height,
		     height_inc, num_heights, kp, doppler_flag, 'sami3', sami_options)


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
# print (ray_path_data)


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





print('\n')
plt.show()
