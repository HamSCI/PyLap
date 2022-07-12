#
# Name :
#   ray_test2.m
#
# Purpose :
#   Example of using raytrace_2d for a single ray. Non-default input ray state
#   vector examples are also included. IRI2016 used to generate the ionosphere.
#
# Calling sequence :
#   ray_test2
#
# Inputs :
#   None
#
# Outputs :
#   None
#
# Author:
#   V1.0  M.A. Cervera  16/06/2006
#
#   V1.1  M.A. Cervera  28/09/2006  
#     Minor update against raytrace_2d
#
#   V2.0  M.A. Cervera  15/06/2007  
#     Updated for Pharlap V2. Extra raytracing calls to exemplify the new ray 
#     state vector features of raytrace_2d.
#
#   V2.1 M.A. Cervera  12/03/2008
#      Minor update against raytrace_2d. 
#
#   V2.2 M.A. Cervera  01/05/2008
#      Renamed to ray_test2 and modified to work with updated raytrace_2d
#      (for pre-release version of PHaRLAP 3.0)
#
#   V2.3 M.A. Cervera  15/12/2008
#      Now uses IRI2007 to generate the ionosphere
#
#   V2.4 M.A. Cervera  19/05/2011
#      More efficient handling of ionospheric grids in call to raytrace_2d 
#
#   V2.5  M.A. Cervera  02/05/2016
#      Updated to use IRI2016
#
#   V2.6  M.A. Cervera  20/05/2016
#      Updated to use multi-threaded raytrace_2d
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
UT = [2000, 9, 21, 21, 0]        # UT - year, month, day, hour, minute
speed_of_light = 2.99792458e8
R12 = 100

#freq = 10.0                 # ray frequency (MHz)
freq = np.ones(1, dtype = float)
freq[0] = 10.0
# elev = 10                   # ray elevation
elev = np.ndarray(1, dtype = float)
elev[0] = 10
ray_bear = 329              # bearing of ray 
origin_lat = -20.0          # latitude of the start point of ray
origin_long = 130.0         # longitude of the start point of ray
nhops = 1                   # number of hops
tol = [1e-7, .01, 10]         # ODE tolerance and min/max step sizes
doppler_flag = 1            # not interested in Doppler shift
irregs_flag = 0             # no irregularities
kp = 0                      # kp not used as irregs_flag = 0. Set it to a 
                             # dummy value 

print( ['\n' 
    'Example of 2D numerical raytracing where a non-standard input ray state ' \
    'vector has been used.\n'])
print( 
   ['This example demonstrates how one might modify the ray state vector' \
     ' mid-path and could be\n']) 
print( 
    ['used to simulate reflection from a sporadic-E layer.\n\n'])

#
# generate ionospheric, geomagnetic and irregularity grids
#
max_range = 10000      # maximum range for sampling the ionosphere (km)
num_range = 201        # number of ranges (must be < 2001)
range_inc = max_range / (num_range - 1)  # range cell size (km)

start_height = 60      # start height for ionospheric grid (km)
height_inc = 2         # height increment (km)
num_heights = 300      # number of  heights (must be < 2001)

# clear iri_options
# iri_options.Ne_B0B1_model = 'Bil-2000' # this is a non-standard setting for 
                                        # IRI but is used as an example
iri_options = {
               'Ne_B0B1_model': 'Bil-2000'
              }

tic = time.time() 
print('Generating ionospheric grid... ')
[iono_pf_grid, iono_pf_grid_5, collision_freq, irreg, iono_te_grid] = \
     gen_iono.gen_iono_grid_2d(origin_lat, origin_long, R12, UT, ray_bear, 
                    max_range, num_range, range_inc, start_height, 
		    height_inc, num_heights, kp, doppler_flag, 'iri2016', 
		    iri_options)
toc = time.time()
elasped_time = toc - tic

# convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid = iono_pf_grid**2 / 80.6164e-6
iono_en_grid_5 = iono_pf_grid_5**2 / 80.6164e-6

#
# call raytrace
#
tic = time.time() 
print('Generating 2D NRT ray ...')
[ray_data, ray_path_data, ray_state_vec] = \
      raytrace_2d(origin_lat, origin_long, elev, ray_bear, freq, nhops, tol, 
             irregs_flag, iono_en_grid, iono_en_grid_5, collision_freq, 
	     start_height, height_inc, range_inc, irreg)
	 	 
toc = time.time()

# plot the ray	 
plt.scatter(ray_path_data[0]['ground_range'][0], ray_path_data[0]['height'][0],
        s=None, c='b', marker= 'x', label = 'Original ray', cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None,
        edgecolors=None, plotnonfinite=False, data=None)
for i in range(0,len(ray_path_data[0]['height'])):
    plt.scatter(ray_path_data[0]['ground_range'][i], ray_path_data[0]['height'][i],
        s=None, c='b', marker= 'x', cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None,
        edgecolors=None, plotnonfinite=False, data=None)
plt.xlim(0, 2000)
plt.xlabel('Ground Range (km)')
plt.ylabel('Height (km)')

#
# new raytrace using a non-default input ray state vector - use that from a
# point along the previous raytrace (90 km in altittude) and see if we can
# replicate the previous raytrace 
#
idx = np.argwhere(ray_path_data[0]['height'] > 90)
idx = idx[0]

ray_state_vec_in2 = {'r': np.ndarray([]), 'Q': np.ndarray(1), 'theta': np.ndarray(1), 
                    'delta_r': np.ndarray(1), 'delta_Q': np.ndarray(1), 
                    'absorption': np.ndarray(1),'phase_path': np.ndarray(1), 
                    'group_path': np.ndarray(1), 'group_path_step_size': np.ndarray(1)}
# ray_state_vec_in2 = np.ndarray(9)
ray_state_vec_in2['r'] = ray_state_vec[0]['r'][idx]
ray_state_vec_in2['Q'] = ray_state_vec[0]['Q'][idx]
ray_state_vec_in2['theta'] = ray_state_vec[0]['theta'][idx]
ray_state_vec_in2['delta_r'] = ray_state_vec[0]['delta_r'][idx]
ray_state_vec_in2['delta_Q'] = ray_state_vec[0]['delta_Q'][idx]
ray_state_vec_in2['absorption'] = ray_state_vec[0]['deviative_absorption'][idx]
ray_state_vec_in2['phase_path'] = ray_state_vec[0]['phase_path'][idx]
ray_state_vec_in2['group_path'] = ray_state_vec[0]['group_path'][idx]
ray_state_vec_in2['group_path_step_size'] = ray_state_vec[0]['group_step_size'][idx]

[ray_data2, ray_path_data2, ray_state_vec2] = \
    raytrace_2d(origin_lat, origin_long, elev, ray_bear, freq, nhops, tol, 
                irregs_flag, iono_en_grid, iono_en_grid_5, collision_freq, 
	            start_height, height_inc, range_inc, irreg, ray_state_vec_in2)

# overplot the ray	 
# hold on
plt.scatter(ray_path_data2[0]['ground_range'][0], ray_path_data2[0]['height'][0],
        s=None, c='r', marker= 'x', label = 'Ray unaffected by Es Layer', cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None,
        edgecolors=None, plotnonfinite=False, data=None)
for i in range(0,len(ray_path_data2[0]['height'])):
    plt.scatter(ray_path_data2[0]['ground_range'][i], ray_path_data2[0]['height'][i],
        s=None, c='r', marker= 'x', cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None,
        edgecolors=None, plotnonfinite=False, data=None)
# hold off
#
# 3rd raytrace using a non-default input ray state vector - use that from a
# point along the previous raytrace (90 km in altittude) and see if we can
# reflect the previous raytrace 
#
ray_state_vec_in3 = ray_state_vec_in2
ray_state_vec_in3['Q']= -ray_state_vec_in3['Q']

# tic = time.time() 
print('Generating "reflected" 2D NRT ray ...')
[ray_data3, ray_path_data3, ray_path_data_vec3] = \
    raytrace_2d(origin_lat, origin_long, elev, ray_bear, freq, nhops, 
                tol, irregs_flag, iono_en_grid, iono_en_grid_5, collision_freq, 
	            start_height, height_inc, range_inc, irreg, ray_state_vec_in3)
# toc = time.time() 

# overplot the ray	 
# hold on\
plt.scatter(ray_path_data3[0]['ground_range'][0], ray_path_data3[0]['height'][0],
        s=None, c='g', marker= 'x', label = 'Ray reflected by Es Layer', cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None,
        edgecolors=None, plotnonfinite=False, data=None)
for i in range(0,len(ray_path_data3[0]['height'])):
    plt.scatter(ray_path_data3[0]['ground_range'][i], ray_path_data3[0]['height'][i],
        s=None, c='g', marker= 'x', cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None,
        edgecolors=None, plotnonfinite=False, data=None)
plt.plot([300, 600], [90, 90], 'k')
plt.text(100, 90, 'Es Layer')
# hold off
#
# display the legend
#
plt.legend(loc = 'upper right', prop={'size': 8})   
   
print('\n')   

plt.show()
