#
# Name :
#   ray_test4.m
#
# Purpose :
#   Example of using raytrace_2d for a set of rays with the same launch
#   direction at different frequencies. Ray trajectories are plotted over
#   the ionosphere which has been generated by IRI 2016.
#
# Calling sequence :
#   ray_test4
#
# Inputs :
#   None
#
# Outputs :
#   None
#
# Change log:
#   V1.0  M.A. Cervera  09/06/2016
#     Initial Version
#
import numpy as np  # py
import time
import ctypes as c


from pylap.raytrace_2d import raytrace_2d 
from Ionosphere import gen_iono_grid_2d as gen_iono
from Plotting import plot_ray_iono_slice as plot_iono

#import raytrace_2d as raytrace
import matplotlib.pyplot as plt

plt.switch_backend('tkagg')

#
# setup general stuff
#
UT = [2001, 3, 15, 7, 0]        # UT - year, month, day, hour, minute
R12 = 100                   # R12 index
speed_of_light = 2.99792458e8

ray_bear = 324.7
origin_lat = -23.5          # latitude of the start point of ray
origin_long = 133.7         # longitude of the start point of ray
doppler_flag = 1            # generate ionosphere 5 minutes later so that
                             # Doppler shift can be calculated
irregs_flag = 0             # no irregularities - not interested in 
                             # Doppler spread or field aligned irregularities
kp = 0                      # kp not used as irregs_flag = 0. Set it to a 
                             # dummy value 

print( \
  '\nExample of 2D numerical raytracing for a ray at different frequencies\n\n')

#
# generate ionospheric, geomagnetic and irregularity grids
#
max_range = 10000      # maximum range for sampling the ionosphere (km)
num_range = 201        # number of ranges (must be < 2000)
range_inc = max_range / (num_range - 1)  # range cell size (km)

start_height = 0       # start height for ionospheric grid (km)
height_inc = 3         # height increment (km)
num_heights = 200      # number of  heights (must be < 2000)

# clear iri_options
#iri_options.Ne_B0B1_model = 'Bil-2000'  #M this is a non-standard setting for
                                        #M IRI but is used as an example
# implement the above by means of dictionay
iri_options = {
               'Ne_B0B1_model': 'Bil-2000'
              }

print('Generating ionospheric grid... ')
[iono_pf_grid, iono_pf_grid_5, collision_freq, irreg, iono_te_grid] = \
    gen_iono.gen_iono_grid_2d(origin_lat, origin_long, R12, UT, ray_bear, 
                     max_range, num_range, range_inc, start_height, 
		     height_inc, num_heights, kp, doppler_flag, 'iri2016', 
		     iri_options)
 

# convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid = iono_pf_grid**2 / 80.6164e-6
iono_en_grid_5 = iono_pf_grid_5**2 / 80.6164e-6


#
# Example 1 - Fan of rays, 10 MHz, single hop. Print to encapsulated
# postscript and PNG. Note the transition from E-low to E-High to F2-low modes.
#

# call raytrace for a fan of rays
freq = 10.0         # frequency (MHz)
tol = [1e-7, 0.01, 10]          # ODE tolerance
nhops = 2
ray = []
idx = 1

# first call to raytrace so pass in the ionospheric and geomagnetic grids 
freqs = np.arange(5, 35, dtype = float)
elevs = np.ones(freqs.shape)*20
num_elevs = len(elevs)
tic = time.time() 
print('Generating {} 2D NRT rays ...', num_elevs)
[ray_data, ray_path_data, ray_path_state] = \
    raytrace_2d(origin_lat, origin_long, elevs, ray_bear, freqs, nhops, 
             tol, irregs_flag, iono_en_grid, iono_en_grid_5, 
	     collision_freq, start_height, height_inc, range_inc, irreg)

# plot the rays and ionosphere

################
### Figure 1 ###
################
start_range = 0
start_range_idx = int(start_range/range_inc)
end_range = 2500
end_range_idx = int((end_range) / range_inc) + 1
start_ht = start_height
start_ht_idx = 0
end_ht = 450
end_ht_idx = int(end_ht / height_inc) + 1
iono_pf_subgrid = iono_pf_grid[start_ht_idx:end_ht_idx, start_range_idx:end_range_idx]
ax, ray_handle = plot_iono.plot_ray_iono_slice(iono_pf_subgrid, start_range,
                      end_range, range_inc, start_ht, end_ht, height_inc,
                      ray_path_data,linewidth=1.5, color=[1, 1, 0.99])

fig_str_a = '{}/{}/{}  {:02d}:{:02d}UT   elevation = {} deg   R12 = {}'.format(
              UT[1], UT[2], UT[0], UT[3], UT[4], elevs[1], R12)
fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
             origin_lat, origin_long, ray_bear)

fig_str = fig_str_a + fig_str_b

ax.set_title(fig_str)

print('\n')
plt.show()