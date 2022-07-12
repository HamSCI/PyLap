# 
#  Name :
#    ray_test_3d_pol.m
# 
#  Purpose :
#    Example of using raytrace_3d for a single ray. Plots polarization of wave
#    along the ray path
# 
#  Calling sequence_pol :
#    ray_test_3d
# 
#  Inputs :
#    None
# 
#  Outputs :
#    None
# 
#  Author:
#    V1.0  M.A. Cervera  28/06/2010
# 
#    V1.1 M.A. Cervera  19/05/2011
#       More efficient handling of ionospheric and geomagnetic grids grids in
#       call to raytrace_3d  
# 
#    V2.0 M.A. Cervera  03/05/2016
#     Modified to make use of multi-threaded raytrace_3d. IRI2016 is now used
#     to generate the ionosphere.
# 

import math
from matplotlib import projections
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
UT =[2000, 9, 21, 0, 0]       # UT - year, month, day, hour, minute
speed_of_light = 2.99792458e8
R12 = 100
elev = np.array((25,), dtype = float)              #  % initial ray elevation - two rays
num_elev = len(elev)
freq = 10.0              #   % frequency (MHz)
# freqs = freq * np.ones(num_elev, dtype = float)
ray_bear = np.zeros(len(elev))             #   % initial bearing of  ray
freqs = np.ones(len(elev))*10

origin_lat = -20     #   % latitude of the start point of ray
origin_long = 130     #  % longitude of the start point of ray
origin_ht = 0.0

doppler_flag = 0          #  % not interested in Doppler shift and spread

# %
# % generate ionospheric, geomagnetic and irregularity grids
# %
ht_start = 60          #% start height for ionospheric grid (km)
ht_inc = 2            # % height increment (km)
num_ht = 201          # % number of  heights (must be <= 401)

lat_start = -22
lat_inc = 0.5
num_lat = 101          #% number of latitudes (must be <= 701)

lon_start= 128.0
lon_inc = 1.0
num_lon = 5.0         # % number of longitudes (must be <= 701)
 
iono_grid_parms = [lat_start, lat_inc, num_lat, lon_start, lon_inc, num_lon,
       ht_start, ht_inc, num_ht, ]

B_ht_start = 60; #ht_start      #    % start height for geomagnetic grid (km)
B_ht_inc = 10;                #  % height increment (km)
B_num_ht = math.ceil(num_ht * ht_inc / B_ht_inc)
B_lat_start = lat_start
B_lat_inc = 1.0
B_num_lat = math.ceil(num_lat * lat_inc / B_lat_inc)
B_lon_start = lon_start
B_lon_inc = 1.0
B_num_lon = math.ceil(num_lon * lon_inc / B_lon_inc); 
geomag_grid_parms = [B_lat_start, B_lat_inc, B_num_lat, B_lon_start, 
       B_lon_inc, B_num_lon, B_ht_start, B_ht_inc, B_num_ht]

doppler_flag = 0

tic = time.time()

print('Generating ionospheric and geomag grids... ')
[iono_pf_grid, iono_pf_grid_5, collision_freq, Bx, By, Bz] = \
    gen_iono.gen_iono_grid_3d(UT, R12, iono_grid_parms,
                     geomag_grid_parms, doppler_flag)#all within range except collision_freq

toc = time.time()

#  convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid = iono_pf_grid**2 / 80.6164e-6
iono_en_grid_5 = iono_pf_grid_5**2 / 80.6164e-6



# %
# % call raytrace
# %

nhops = 1        #   % number of hops
tol = [1e-7, 0.01, 25]      #   % rkf tolerance

ray_O = []
ray_X = []
ray_N = []

# %
# % Generate O mode ray
# %
OX_mode = 1
# fprintf('Generating O-mode rays... ')
print("Generating 0-mode rays... ")
tic = time.time()

# % first call to raytrace_3d so pass in the ionosphere
[ray_O, ray_path_O, ray_state_vec_O] = \
    raytrace_3d(origin_lat, origin_long, origin_ht, elev, ray_bear, freqs, 
 	     OX_mode, nhops, tol, iono_en_grid, iono_en_grid_5, 
	     collision_freq, iono_grid_parms, Bx, By, Bz, geomag_grid_parms); 

toc = time.time()


# %
# % Generate X mode ray
# %
OX_mode = -1
# fprintf('Generating X-mode rays... ')
print ("Generating x-mode rays... ")
# % ionosphere already in memory so no need to pass it in again
[ray_X, ray_path_X, ray_state_vec_X] = \
     raytrace_3d(origin_lat, origin_long, origin_ht, elev, ray_bear, freqs, 
	     OX_mode, nhops, tol)
toc = time.time()


# %
# % Generate 'no-field' mode ray
# %
OX_mode = 0
# fprintf('Generating ''no-field'' rays... ')
print ("Generating 'no-field' rays... ")
tic = time.time()

# % ionosphere already in memory so no need to pass it in again
[ray_N, ray_path_N, ray_state_vec_N] = \
     raytrace_3d(origin_lat, origin_long, origin_ht, elev, ray_bear, freqs, 
	     OX_mode, nhops, tol)

toc = time.time()

# fprintf('\n')
print('\n')
# % ionosphere is no longer needed by ratrace_3d so clear it.
# clear raytrace_3d


####################
### First figure ###
####################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
symsize = 5
ax.scatter(ray_path_O[0]['lat'][0], np.mod(ray_path_O[0]['lon'][0],360), ray_path_O[0]['height'][0], zdir='z', c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.', s=symsize)
ax.scatter(ray_path_X[0]['lat'][0], np.mod(ray_path_X[0]['lon'][0],360), ray_path_X[0]['height'][0], zdir='z', c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.', s=symsize)
ax.scatter(ray_path_N[0]['lat'][0], np.mod(ray_path_N[0]['lon'][0],360), ray_path_N[0]['height'][0], zdir='z', c= 'g', label = 'No Mag-field', linewidth=5, linestyle=':', marker='.', s=symsize)

for i in range(0,num_elev,2):
       ax.scatter(ray_path_O[i]['lat'], np.mod(ray_path_O[i]['lon'],360), ray_path_O[i]['height'], zdir='z', c= 'b', linewidth=0.3, linestyle=':', marker='.', s=symsize)
       ax.scatter(ray_path_X[i]['lat'], np.mod(ray_path_X[i]['lon'],360), ray_path_X[i]['height'], zdir='z', c= 'r', linewidth=0.3, linestyle=':', marker='.', s=symsize)
       ax.scatter(ray_path_N[i]['lat'], np.mod(ray_path_N[i]['lon'],360), ray_path_N[i]['height'], zdir='z', c= 'g',  linewidth=0.3, linestyle=':', marker='.', s=symsize)

ax.set_title("3d pol Raytrace")
leg = plt.legend(loc='upper right', shadow=True, fontsize='large')
ax.set_xlabel('Latitude(deg)')
ax.set_ylabel('Longitude(deg)')
ax.set_zlabel('Height (km)')




#####################

### Second figure ###
#####################

fig = plt.figure(figsize=(9, 9), dpi=80)
ax = fig.subplots(2,2)
##################
### First Plot ###
##################

### set the lables for the legend ### 

ax[0,0].scatter(ray_path_O[0]['group_range'][0], np.mod(ray_path_O[0]['polariz_mag'][0],360),  c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.')
ax[0,0].scatter(ray_path_X[0]['group_range'][0], np.mod(ray_path_X[0]['polariz_mag'][0],360),  c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.')

### plot the actual graph ###
ax[0,0].scatter(ray_path_O[0]['group_range'], np.mod(ray_path_O[0]['polariz_mag'],360),  c= 'b', linewidth=1, linestyle=':', marker='.')
ax[0,0].scatter(ray_path_X[0]['group_range'], np.mod(ray_path_X[0]['polariz_mag'],360),  c= 'r', linewidth=1, linestyle=':', marker='.')

### setting labels and legend settings ###
ax[0,0].set_title('Wave E-field vector axial ratio ')
leg = plt.legend(loc='best', shadow=True, fontsize='large')
ax[0,0].set_xlabel('Group Range (km)')
ax[0,0].set_ylabel('Polarization,|R|')


###################
### Second Plot ###
###################

### set the lables for the legend ### 
ax[0,1].scatter(ray_path_O[0]['group_range'][0], np.mod(ray_path_O[0]['wave_Efield_tilt'][0],360),  c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.')
ax[0,1].scatter(ray_path_X[0]['group_range'][0], np.mod(ray_path_X[0]['wave_Efield_tilt'][0],360),  c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.')


### plot the actual graph ###
ax[0,1].scatter(ray_path_O[0]['group_range'], np.mod(ray_path_O[0]['wave_Efield_tilt'],360),  c= 'b', linewidth=1, linestyle=':', marker='.')
ax[0,1].scatter(ray_path_X[0]['group_range'], (ray_path_X[0]['wave_Efield_tilt']),  c= 'r', linewidth=1, linestyle=':', marker='.')

# print(ray_path_X[0]['wave_Efield_tilt'])
### setting labels and legend settings ###
ax[0,1].set_title('Wave E-field vector tilt angle ')
leg = plt.legend(loc='best', shadow=True, fontsize='large')
ax[0,1].set_xlabel('Group Range (km)')
ax[0,1].set_ylabel('Polarization, psi (degrees)')

##################
### Third Plot ###
##################

### set the lables for the legend ### 
ax[1,0].scatter(ray_path_O[0]['group_range'][0], np.mod(ray_path_O[0]['wavenorm_B_angle'][0],360),  c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.')
ax[1,0].scatter(ray_path_X[0]['group_range'][0], np.mod(ray_path_X[0]['wavenorm_B_angle'][0],360),  c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.')

### plot the actual graph ###
ax[1,0].scatter(ray_path_O[0]['group_range'], np.mod(ray_path_O[0]['wavenorm_B_angle'],360),  c= 'b', linewidth=1, linestyle=':', marker='.')
ax[1,0].scatter(ray_path_X[0]['group_range'], np.mod(ray_path_X[0]['wavenorm_B_angle'],360),  c= 'r', linewidth=1, linestyle=':', marker='.')

### setting labels and legend settings ###
ax[1,0].set_title('Angle between wave-normal and B-field')
leg = plt.legend(loc='best', shadow=True, fontsize='large')
ax[1,0].set_xlabel('Group Range (km)')
ax[1,0].set_ylabel('theta (degrees)')

###################
### fourth Plot ###
################### 

### set the lables for the legend ### 
ax[1,1].scatter(ray_path_O[0]['group_range'][0], np.mod(ray_path_O[0]['wavenorm_ray_angle'][0],360),  c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.')
ax[1,1].scatter(ray_path_X[0]['group_range'][0], np.mod(ray_path_X[0]['wavenorm_ray_angle'][0],360),  c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.')

### plot the actual graph ###
ax[1,1].scatter(ray_path_O[0]['group_range'], np.mod(ray_path_O[0]['wavenorm_ray_angle'],360),  c= 'b', linewidth=1, linestyle=':', marker='.')
ax[1,1].scatter(ray_path_X[0]['group_range'], np.mod(ray_path_X[0]['wavenorm_ray_angle'],360),  c= 'r', linewidth=1, linestyle=':', marker='.')

### setting labels and legend settings ###
ax[1,1].set_title('Angle between wave-normal and ray direction')
leg = plt.legend(loc='best', shadow=True, fontsize='large')
ax[1,1].set_xlabel('Group Range (km)')
ax[1,1].set_ylabel('alpha (degrees)')



#########################################
### set the spacing between the plots ###
#########################################
# plt.subplots_adjust(
#                     wspace=2, 
#                     hspace=.5)


################
### Figure 3 ###
################

fig, (ax1, ax2) = plt.subplots(1,2)


##############
### Plot 1 ###
##############

### set the lables for the legend ### 
ax1.scatter(ray_path_O[0]['group_range'][0], np.mod(ray_path_O[0]['refractive_index'][0],360),  c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.')
ax1.scatter(ray_path_X[0]['group_range'][0], np.mod(ray_path_X[0]['refractive_index'][0],360),  c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.')



### plot the actual graph ###
ax1.scatter(ray_path_O[0]['group_range'], np.mod(ray_path_O[0]['refractive_index'],360),  c= 'b', linewidth=1, linestyle=':', marker='.')
ax1.scatter(ray_path_X[0]['group_range'], np.mod(ray_path_X[0]['refractive_index'],360),  c= 'r', linewidth=1, linestyle=':', marker='.')

### setting labels and legend settings ###
ax1.set_title('Refractive Index')
leg = plt.legend(loc='best', shadow=True, fontsize='large')
ax1.set_xlabel('Group Range (km)')
ax1.set_ylabel('Polarization, psi (degrees)')

##############
### Plot 2 ###
##############

### set the lables for the legend ### 
ax2.scatter(ray_path_O[0]['group_range'][0], np.mod(ray_path_O[0]['group_refractive_index'][0],360),  c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.')
ax2.scatter(ray_path_X[0]['group_range'][0], np.mod(ray_path_X[0]['group_refractive_index'][0],360),  c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.')



### plot the actual graph ###
ax2.scatter(ray_path_O[0]['group_range'], np.mod(ray_path_O[0]['group_refractive_index'],360),  c= 'b', linewidth=1, linestyle=':', marker='.')
ax2.scatter(ray_path_X[0]['group_range'], np.mod(ray_path_X[0]['group_refractive_index'],360),  c= 'r', linewidth=1, linestyle=':', marker='.')


### set the lables for the legend ### 
ax2.set_title('Group Refractive Index')
leg = plt.legend(loc='best', shadow=True, fontsize='large')
ax2.set_xlabel('Group Range (km)')
ax2.set_ylabel('Polarization, psi (degrees)')
plt.show()

