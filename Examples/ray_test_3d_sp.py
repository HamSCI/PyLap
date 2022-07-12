#
# Name :
#   ray_test_3d_sp.m
#
# Purpose :
#   Example of using raytrace_3d_sp for a fan of rays. 
#
# Calling sequence :
#   ray_test_3d_sp
#
# Inputs :
#   None
#
# Outputs :
#   None
#
# Modification History:
#   V1.0  M.A. Cervera  22/04/2015
#     Initial version branched from raytrace_3d (V1.2)
#
#   V2.0  M.A. Cervera  03/05/2016
#     Modified to make use of multi-threaded raytrace_3d. IRI2016 is now used
#     to generate the ionosphere.
#

import math
import numpy as np  # py
import time
import ctypes as c
#import plot_ray_iono_slice as plot_iono


from Ionosphere import gen_iono_grid_3d as gen_iono
from pylap.raytrace_3d import raytrace_3d
from Plotting import plot_ray_iono_slice as plot_ray_iono_slice
from Maths import raz2latlon
from Maths import latlon2raz
#import raytrace_2d as raytrace
import matplotlib.pyplot as plt


#
# setup general stuff
#
UT = [2000, 9, 21, 0, 0]            # UT - year, month, day, hour, minute
speed_of_light = 2.99792458e8 
re = np.array((6376000,), dtype = float)                    # radius of Earth to use for coord conv.
R12 = 100 
elevs = np.arange(3,81,1, dtype=float)             # initial ray elevation
num_elevs = len(elevs)
freqs = np.ones(num_elevs)*15    # frequency (MHz)
ray_bears = np.zeros(num_elevs)  # initial bearing of rays
origin_lat = -20.0              # latitude of the start point of ray
origin_long = 130.0             # longitude of the start point of ray
origin_ht = 0.0 
doppler_flag = 1                # interested in Doppler shift


print('\nExample of 3D magneto-ionic numerical raytracing for a spherical Earth\n' 
'Earth radius used is #f km\n\n')

#
# generate ionospheric, geomagnetic and irregularity grids
#
ht_start = 60           # start height for ionospheric grid (km)
ht_inc = 2              # height increment (km)
num_ht = 201            # number of heights (must be <= 401)
lat_start = -20.0 
lat_inc = 0.3 
num_lat = 101.0         # number of latitudes (must be <= 701)
lon_start= 128.0 
lon_inc = 1.0 
num_lon = 5.0           # number of longitudes (must be <= 701)
iono_grid_parms = [lat_start, lat_inc, num_lat, lon_start, lon_inc, num_lon, 
      ht_start, ht_inc, num_ht, ] 

# iono_grid_parms = np.array((lat_start, lat_inc, num_lat, lon_start, lon_inc, num_lon, 
#       ht_start, ht_inc, num_ht,),dtype = float)
B_ht_start = ht_start           # start height for geomagnetic grid (km)
B_ht_inc = 10                   # height increment (km)
B_num_ht = math.ceil(num_ht * ht_inc / B_ht_inc) 
B_lat_start = lat_start 
B_lat_inc = 1.0 
B_num_lat = math.ceil(num_lat * lat_inc / B_lat_inc) 
B_lon_start = lon_start 
B_lon_inc = 1.0 
B_num_lon = math.ceil(num_lon * lon_inc / B_lon_inc)  
geomag_grid_parms = [B_lat_start, B_lat_inc, B_num_lat, B_lon_start, 
      B_lon_inc, B_num_lon, B_ht_start, B_ht_inc, B_num_ht] 


# tic
print('Generating ionospheric and geomag grids... ')
[iono_pf_grid, iono_pf_grid_5, collision_freq, Bx, By, Bz] = \
    gen_iono.gen_iono_grid_3d(UT, R12, iono_grid_parms, 
                     geomag_grid_parms, doppler_flag, 'iri2016') 
# toc
print('\n')
# geomag_grid_parms = np.array((B_lat_start, B_lat_inc, B_num_lat, B_lon_start, 
#       B_lon_inc, B_num_lon, B_ht_start, B_ht_inc, B_num_ht,), dtype = float)
# convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid = iono_pf_grid**2 / 80.6164e-6 
iono_en_grid_5 = iono_pf_grid_5**2 / 80.6164e-6 

#
# call raytrace
#
nhops = 4                   # number of hops
tol = [1e-7, 0.01, 25]        # ODE solver tolerance and min max stepsizes
num_elevs = len(elevs) 

# Generate the O mode rays
OX_mode = 1 
print('Generating O-mode rays... ')
# tic
[ray_data_O, ray_O, ray_state_vec_O] = \
  raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, 
              freqs, OX_mode, nhops, tol, iono_en_grid, iono_en_grid_5, 
	      collision_freq, iono_grid_parms, Bx, By, Bz, 
	      geomag_grid_parms) 
# NRT_total_time = toc 
# fprintf('\n   NRT-only execution time = #f, Total mex execution time = #f\n\n', ...
        # [ray_data_O.NRT_elapsed_time], NRT_total_time)

for rayId in range(0, num_elevs):
  num = len(ray_O[rayId]['lat']) 
  ground_range = np.zeros((2, num)) 
  lat = ray_O[rayId]['lat'] 
  lon = ray_O[rayId]['lon']  
  ground_range[0:num] = latlon2raz.latlon2raz(lat[0:num], lon[0:num], origin_lat, 
      origin_long,'wgs84')  #/1000.0 
  ground_range = ground_range/1000.0

  ray_O[rayId]['ground_range'] = ground_range[0]



# Generate the X mode rays - note in the raytrace_3d call the ionosphere does
# not need to be passed in again as it is already in memory
OX_mode = -1 
 
print('Generating #d X-mode rays ...', num_elevs) 
# tic
[ray_data_X, ray_X, ray_sv_X] = \
    raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, 
                 freqs, OX_mode, nhops, tol) 
# NRT_total_time = toc 
# fprintf('\n   NRT-only execution time = #f, Total mex execution time = #f\n\n', \
#         [ray_data_X.NRT_elapsed_time], NRT_total_time)

for rayId in range(0, num_elevs):
    num = len(ray_X[rayId]['lat']) 
    ground_range = np.zeros((2, num)) 
    lat = ray_X[rayId]['lat']
    lon = ray_X[rayId]['lon']     
    ground_range[0:num] = latlon2raz.latlon2raz(lat[0:num], lon[0:num], origin_lat, 
       origin_long,'wgs84')#/1000.0 
    ground_range = ground_range/1000.0
    ray_X[rayId]['ground_range'] = ground_range[0]



# Generate the rays for the case where the magnetic field is ignored
OX_mode = 0 

print('Generating ''no-field'' rays... ')

# tic
[ray_data_N, ray_N, ray_sv_N] = \
  raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, 
                 freqs, OX_mode, nhops, tol) 
# NRT_total_time = toc 
# fprintf('\n   NRT-only execution time = #f, Total mex execution time = #f\n\n', ...
#         [ray_data_N.NRT_elapsed_time], NRT_total_time)

for rayId in range (1, num_elevs):
  num = len(ray_N[rayId]['lat']) 
  ground_range = np.zeros((2, num)) 
  lat = ray_N[rayId]['lat'] 
  lon = ray_N[rayId]['lon']     
  ground_range[0:num] = latlon2raz.latlon2raz(lat[0:num], lon[0:num], origin_lat, 
      origin_long,'wgs84')#/1000.0 
ground_range = ground_range/1000.0
ray_N[rayId]['ground_range'] = ground_range[0]


print('\n')





# plot the rays


####################
### First figure ###
####################
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
symsize = 5

### set the lables for the legend ### 
ax.scatter(ray_O[0]['lat'][0], np.mod(ray_O[0]['lon'][0], 360), ray_O[0]['height'][0],
           zdir='z', c='b', label='O Mode', linewidth=7, linestyle=':', marker='.', s=symsize)
ax.scatter(ray_X[0]['lat'][0], np.mod(ray_X[0]['lon'][0], 360), ray_X[0]['height'][0],
           zdir='z', c='r', label='X Mode', linewidth=7, linestyle=':', marker='.', s=symsize)
ax.scatter(ray_N[0]['lat'][0], np.mod(ray_N[0]['lon'][0], 360), ray_N[0]['height'][0],
           zdir='z', c='g', label='No Mag-field', linewidth=7, linestyle=':', marker='.', s=symsize)

### plot the actual graph ###
for i in range (0, num_elevs, 2):
    ax.scatter(ray_O[i]['lat'], np.mod(ray_O[i]['lon'], 360), ray_O[i]['height'],
           zdir='z', c='b',  linewidth=0.3, linestyle=':', marker='.', s=symsize)
    ax.scatter(ray_X[i]['lat'], np.mod(ray_X[i]['lon'], 360), ray_X[i]['height'],
           zdir='z', c='r',  linewidth=0.3, linestyle=':', marker='.', s=symsize)
    ax.scatter(ray_N[i]['lat'], np.mod(ray_N[i]['lon'], 360), ray_N[i]['height'],
           zdir='z', c='g', linewidth=0.3, linestyle=':', marker='.', s=symsize)

### setting labels and legend settings ###
ax.set_title("3d Raytrace SP")
leg = plt.legend(loc='upper right', shadow=True, fontsize='large')
# get the lines and texts inside legend box
ax.set_xlabel('Latitude (deg)')
ax.set_ylabel('Longitude (deg)')
ax.set_zlabel('Height (km)')





#####################
### Second figure ###
##################### 

start_range = 0 
end_range = 2000 
range_inc = 50 
end_range_idx = np.fix((end_range-start_range) / range_inc) + 1 
start_ht = 0 
start_ht_idx = 1 
height_inc = 5 
end_ht = 350 
end_ht_idx = np.fix(end_ht / height_inc) + 1 
iono_pf_subgrid = np.zeros((int(end_ht_idx), int(end_range_idx))) 
freq = 15.0
ax, handle = plot_ray_iono_slice.plot_ray_iono_slice(iono_pf_subgrid, start_range, end_range, range_inc, 
    start_ht, end_ht, height_inc, ray_O, linewidth = 1.5, color = [1, 1, 0.99]) 

fig_str_a = '{}/{}/{}  {:02d}:{:02d}UT   {}MHz   R12 = {}'.format(
    UT[1], UT[2], UT[0], UT[3], UT[4], freq, R12)
fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
    origin_lat, origin_long, ray_bears[0])

fig_str = fig_str_a + fig_str_b

ax.set_title(fig_str)

plt.show()