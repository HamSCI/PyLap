 #
 # Name :
 #   ray_test_3d_pol_coupling.m
 #
 # Purpose :
 #   Example of polarization coupling of radio waves emmitted by a vertically 
 #   polarised into the O and X propagtion modes. 
 #
 # Calling sequence :
 #   ray_test_3d_pol_coupling
 #
 # Inputs :
 #   None
 #
 # Outputs :
 #   None
 #
 # Modification History:
 #   V1.0  M.A. Cervera  02/10/2020
 #     Initial version.
 #


 #
 # setup general stuff
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
from Maths import relaz2xyz
from Maths import latlon2raz
from Maths import pol_power_coupling
#import raytrace_2d as raytrace
import matplotlib.pyplot as plt


UT = [2000, 9, 21, 0, 0]              # UT - year, month, day, hour, minute
speed_of_light = 2.99792458e8  
R12 = 100  
elevs = np.arange(0.1, 90, 1, dtype=float)      #[0.1:1:90]              # initial elevation of rays
freqs = np.ones(len(elevs))*15      # frequency (MHz)
origin_ht = 0.0                   # altitude of the start point of rays
doppler_flag = 1                  # interested in Doppler shift


 # choose the case to model
print('Enter the choice to model :\n')
print('  [1] - Southward propagation from Kalkirindji, NT, Australia\n')
print('  [2] - Northward propagation from Kalkirindji, NT, Australia\n')
print('  [3] - Northward propagation from magnetic equator\n')
print('  [4] - Eastward propagation from magnetic equator\n')
choice_str = input('  > ')  
choice = int(choice_str)
print('\n')


if (choice ==1):
     ### case 1 ### 
     ### Southward propagation at Kalkirindji, NT, Australia ###
    ray_bears = np.zeros(len(elevs))+184      # initial bearing of rays
    origin_lat = -17                        # latitude of the start point of rays
    origin_long = 131.0                     # longitude of the start point of rays
elif(choice == 2):
     ### case 2 ###
     ### Northward propagation at Kalkirindji, NT, Australia ###
    ray_bears = np.zeros(len(elevs))+4        # initial bearing of rays
    origin_lat = -17                        # latitude of the start point of rays
    origin_long = 131.0 
elif(choice == 3):
  #   case 2
     # Northward propagation at Kalkirindji, NT, Australia
    ray_bears = np.zeros(len(elevs))+4        # initial bearing of rays
    origin_lat = -17                        # latitude of the start point of rays
    origin_long = 131.0   
elif(choice == 4):
  #   case 4
     # Eastward propagation at magnetic equator
    ray_bears = np.zeros(len(elevs)) + 90     # initial bearing of rays
    origin_lat = 8                          # latitude of the start point of rays
    origin_long = 131.0     
else:
  print('Error invalid input\n Please enter one of the above cases')
# match choice:
#   case 1 
#      # Southward propagation at Kalkirindji, NT, Australia
#     ray_bears = zeros(size(elevs))+184      # initial bearing of rays
#     origin_lat = -17                        # latitude of the start point of rays
#     origin_long = 131.0                     # longitude of the start point of rays

#   case 2
#      # Northward propagation at Kalkirindji, NT, Australia
#     ray_bears = zeros(size(elevs))+4        # initial bearing of rays
#     origin_lat = -17                        # latitude of the start point of rays
#     origin_long = 131.0                     # longitude of the start point of rays

#   case 3 
#      # Northward propagation at magnetic equator
#     ray_bears = zeros(size(elevs))          # initial bearing of rays
#     origin_lat = 8                          # latitude of the start point of rays
#     origin_long = 131.0                     # longitude of the start point of rays

#   case 4
#      # Eastward propagation at magnetic equator
#     ray_bears = zeros(size(elevs)) + 90     # initial bearing of rays
#     origin_lat = 8                          # latitude of the start point of rays
#     origin_long = 131.0                     # longitude of the start point of rays

#   otherwise
#     error('Unknown case')
    
# end


 #
 # generate ionospheric, geomagnetic and irregularity grids
 #
ht_start = 60             # start height for ionospheric grid (km)
ht_inc = 2                # height increment (km)
num_ht = 201             
lat_start = -30.0  
lat_inc = 0.5  
num_lat = 101.0  
lon_start= 130.0  
lon_inc = 1.0  
num_lon = 15.0  
iono_grid_parms = [lat_start, lat_inc, num_lat, lon_start, lon_inc, num_lon, 
      ht_start, ht_inc, num_ht, ]  

B_ht_start = ht_start             # start height for geomagnetic grid (km)
B_ht_inc = 10                     # height increment (km)
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
                     geomag_grid_parms, doppler_flag)  
# toc
print('\n')

 # convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid = iono_pf_grid**2 / 80.6164e-6  
iono_en_grid_5 = iono_pf_grid_5**2 / 80.6164e-6  


 #
 # call raytrace
 #
nhops = 1                     # number of hops
tol = [1e-7, 0.01, 25]          # ODE solver tolerance and min max stepsizes
num_elevs = len(elevs)  

 # Generate the O mode rays
OX_mode = 1  
 
print('Generating  #d O-mode rays ...', num_elevs)  
# tic
[ray_data_O, ray_O, ray_state_vec_O] = \
  raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, freqs, 
              OX_mode, nhops, tol, iono_en_grid, iono_en_grid_5, 
	          collision_freq, iono_grid_parms, Bx, By, Bz, 
	          geomag_grid_parms)  
	      
# NRT_total_time = toc  
  # print('\n   NRT-only execution time =  #f, Total mex execution time =  #f\n\n', ...
        # [ray_data_O.NRT_elapsed_time], NRT_total_time)

for rayId in range(1, num_elevs):
  num = len(ray_O[rayId]['lat'])  
  ground_range = np.zeros((2, num))  
  lat = ray_O[rayId]['lat']  
  lon = ray_O[rayId]['lon']   
  ground_range[0:num] = latlon2raz.latlon2raz(lat[0:num], lon[0:num], origin_lat, 
      origin_long,'wgs84') #/1000.0  
  ground_range = ground_range/1000.0
  ray_O[rayId]['ground_range'] = ground_range[0]  
    		       
   # Calculate incident polarization axial ratio and major axis direction
   # immediately prior to entering the ionosphere. This will actually be the
   # polarization of the radio wave immediately after transmission. 
  axial_ratio_incident = 0                                # linear polarization  
  [antenna_dir_x, antenna_dir_y, antenna_dir_z] =  relaz2xyz.relaz2xyz(1, 90, ray_bears, origin_lat, origin_long)  
  antenna_dir = [antenna_dir_x, antenna_dir_y, antenna_dir_z]  
  [init_wave_dir_x, init_wave_dir_y, init_wave_dir_z] = \
	relaz2xyz.relaz2xyz(1, elevs, ray_bears , origin_lat, origin_long)  
  init_wave_dir = [init_wave_dir_x, init_wave_dir_y, init_wave_dir_z] 
 
  temp = np.cross(np.cross(init_wave_dir, antenna_dir, axis = 0), init_wave_dir, axis = 0)  
  init_pol_dir = temp / np.sqrt(sum(temp**2))  

   # find index where ray is incident on ionosphere
  idx = np.argwhere(ray_O[rayId]['electron_density'] != 0)  
  idx = idx[1]  
  
   # calculate induced polarization axial ratio and major axis direction in
   # the ionosphere
  polariz_mag = ray_O[rayId]['polariz_mag'][idx]  
  if polariz_mag > 1:
    axial_ratio_induced = 1 / polariz_mag  
  else :
    axial_ratio_induced = polariz_mag  
  
  iono_wave_dir = [ray_state_vec_O[rayId]['dir_x'][idx], 
	           ray_state_vec_O[rayId]['dir_y'][idx], 
		   ray_state_vec_O[rayId]['dir_z'][idx]]  
  mag_field_dir = [ray_O[rayId]['geomag_x'][idx], ray_O[rayId]['geomag_y'][idx], 
                   ray_O[rayId]['geomag_z'][idx]]  
 
  mag_field_dir = mag_field_dir / np.sqrt(np.sum(np.power(mag_field_dir, 2))) 
     
  O_pol_dir = np.cross(np.cross(iono_wave_dir, mag_field_dir, axis = 0), iono_wave_dir, axis = 0)  
  O_pol_dir = O_pol_dir / np.sqrt(sum(O_pol_dir**2))  
	     
   # calculate the coupling factor
  cos_polariz_angle_change = np.dot(init_pol_dir, O_pol_dir)  
  
  polariz_angle_change = np.arccos(cos_polariz_angle_change)  	    
  ray_O[rayId]['coupling_factor'] = pol_power_coupling.pol_power_coupling(axial_ratio_incident, 
                                       axial_ratio_induced, polariz_angle_change)  

   # calculate angle between the inicident major axis direction and the
   # magnetic field and the incident wave direction and magnetic field
  cos_init_pol_magfield_angle = np.dot(init_pol_dir, mag_field_dir)  
  ray_O[rayId]['init_pol_magfield_angle'] = np.arccos(cos_init_pol_magfield_angle)  
 
  cos_init_wavedir_magfield_angle = np.dot(init_wave_dir, mag_field_dir)  
  ray_O[rayId]['init_wavedir_magfield_angle'] = \
                           np.arccos(cos_init_wavedir_magfield_angle)  
  




 # Generate the X mode rays - note in the raytrace_3d call the ionosphere does
 # not need to be passed in again as it is already in memory
OX_mode = -1  
 
print('Generating  #d X-mode rays ...', num_elevs)  
# tic
[ray_data_X, ray_X, ray_sv_X] = \
  raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, freqs, 
              OX_mode, nhops, tol)  
# NRT_total_time = toc  
#   print('\n   NRT-only execution time =  #f, Total mex execution time =  #f\n\n', 
#         [ray_data_X.NRT_elapsed_time], NRT_total_time)

for rayId in range(0,num_elevs):
  num = len(ray_X[rayId]['lat'])  
  ground_range = np.zeros((2, num))  
  lat = ray_X[rayId]['lat']
  lon = ray_X[rayId]['lon']     
  ground_range[0:num] = latlon2raz.latlon2raz(lat[0:num], lon[0:num], origin_lat, 
      origin_long,'wgs84')   #/1000.0 
  ground_range = ground_range/1000.0 
  ray_X[rayId]['ground_range'] = ground_range[0] 
  
   # Calculate incident polarization axial ratio and major axis direction
   # immediately prior to entering the ionosphere. This will actually be the
   # polarization of the radio wave immediately after transmission. 
  axial_ratio_incident = 0                                # linear polarization  
  [antenna_dir_x, antenna_dir_y, antenna_dir_z] =   antenna_dir = [antenna_dir_x, antenna_dir_y, antenna_dir_z]  # vpol antennarelaz2xyz.relaz2xyz(1, 90, ray_bears(rayId), origin_lat, origin_long)
  [init_wave_dir_x, init_wave_dir_y, init_wave_dir_z] = \
	relaz2xyz.relaz2xyz(1, elevs, ray_bears, origin_lat, origin_long)  
  init_wave_dir = [init_wave_dir_x, init_wave_dir_y, init_wave_dir_z]  
  temp = np.cross(np.cross(init_wave_dir, antenna_dir), init_wave_dir)  
  init_pol_dir = temp / np.sqrt(sum(temp**2))  

   # find index where ray is incident on ionosphere
  idx = np.argwhere(ray_X[rayId]['electron_density'] != 0)  
  idx = idx[1]  
		       
   # calculate the induced polarization axial ratio and major axis direction
   # in the ionosphere
  polariz_mag = ray_X[rayId]['polariz_mag'][idx]  
  if polariz_mag > 1:
    axial_ratio_induced = 1 / polariz_mag  
  else:
    axial_ratio_induced = polariz_mag  
  
  iono_wave_dir = [ray_sv_X[rayId]['dir_x'][idx], ray_sv_X[rayId]['dir_y'][idx], 
		   ray_sv_X[rayId]['dir_z'][idx]]  
  mag_field_dir = [ray_X[rayId]['geomag_x'][idx], ray_X[rayId]['geomag_y'][idx], 
                   ray_X[rayId]['geomag_z'][idx]]  
  tmp_dir = np.cross(np.cross(iono_wave_dir, mag_field_dir, axis = 0), iono_wave_dir, axis = 0)  
  X_pol_dir = np.cross(iono_wave_dir, tmp_dir, axis = 0)  
  X_pol_dir = X_pol_dir / np.sqrt(sum(X_pol_dir**2))  
  
   # calculate the coupling factor
  cos_polariz_angle_change = np.dot(init_pol_dir, X_pol_dir)  
  polariz_angle_change = np.arccos(cos_polariz_angle_change)  	    
  ray_X[rayId]['coupling_factor'] = pol_power_coupling.pol_power_coupling(axial_ratio_incident, 
                                       axial_ratio_induced, polariz_angle_change)  





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
symsize = 5
# ax.scatter(ray_O[0]['lat'][0], np.mod(ray_O[0]['lon'][0],360), ray_O[0]['height'][0], zdir='z', c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.', s=symsize)
# ax.scatter(ray_X[0]['lat'][0], np.mod(ray_X[0]['lon'][0],360), ray_X[0]['height'][0], zdir='z', c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.', s=symsize)

for i in range(0,len(elevs),2):
       ax.scatter(ray_O[i]['lat'], np.mod(ray_O[i]['lon'],360), ray_O[i]['height'], zdir='z', c= 'b', linewidth=0.3, linestyle=':', marker='.', s=symsize)
       ax.scatter(ray_X[i]['lat'], np.mod(ray_X[i]['lon'],360), ray_X[i]['height'], zdir='z', c= 'r', linewidth=0.3, linestyle=':', marker='.', s=symsize)


ax.set_title("3d pol Coupling Raytrace")
leg = plt.legend(loc='upper right', shadow=True, fontsize='large')
ax.set_xlabel('Latitude(deg)')
ax.set_ylabel('Longitude(deg)')
ax.set_zlabel('Height (km)')




#  # plot the rays
# figure(1)
# plot3(ray_O(1).lat, mod(ray_O(1).lon, 360), ray_O(1).height, '.b', ...
#       'markersize', 5)
# set(gca, 'Zlim', [0 500])
# hold on
# plot3(ray_X(1).lat,  mod(ray_X(1).lon, 360), ray_X(1).height, '.r',  ...
#       'markersize',5)
# for ii = 3:2:num_elevs
#   plot3(ray_O(ii).lat, mod(ray_O(ii).lon, 360), ray_O(ii).height, '.b', ...
#         'markersize', 5)
#   plot3(ray_X(ii).lat, mod(ray_X(ii).lon, 360), ray_X(ii).height, '.r', ...
#         'markersize', 5)
# end  
# hold off
# grid on
# xlabel('latitude (deg)')
# ylabel('longitude (deg)')
# zlabel('Height (km)')
# legend('O Mode', 'X Mode')


fig = plt.figure()
ax = fig.add_subplot(111)

# ax.scatter(elevs, np.mod(ray_O[0]['coupling_factor'],360),  c= 'b', label = 'O Mode',linewidth=5, linestyle=':', marker='.')
# ax.scatter(elevs, np.mod(ray_X[0]['coupling_factor'],360),  c= 'r', label= 'X Mode',linewidth=5, linestyle=':', marker='.')
# for i in range(0,len(elevs),2):
ax.scatter(elevs, np.mod(ray_O[1]['coupling_factor']*100,360),  c= 'b', linewidth=.3, linestyle=':', marker='.')
ax.scatter(elevs, np.mod(ray_X[1]['coupling_factor']*100,360),  c= 'r', linewidth=.3, linestyle=':', marker='.')

ax.set_title('Wave E-field vector tilt angle ')
leg = plt.legend(loc='best', shadow=True, fontsize='large')
ax.set_xlabel('Transmit Elevation (degrees)')
ax.set_ylabel('Coupling factor ( #)')


plt.show()
#  # plot the polarization coupling
# figure(2)
# plot(elevs,[ray_O.coupling_factor]*100, 'b', 'linewidth', 2)
# hold on
# plot(elevs,[ray_X.coupling_factor]*100, 'r', 'linewidth', 2)
# grid on
# set(gca, 'linewidth', 2, 'fontsize', 12, 'xlim', [0 90], 'ylim', [0 100])
# xlabel('Transmit Elevation (degrees)', 'fontsize', 12)
# ylabel('Coupling factor ( #)', 'fontsize', 12)
# title('Vertically polarized transmit antenna', 'fontsize', 12)
# lh = legend('O Mode', 'X Mode')  
# set(lh, 'location', 'southwest')
# hold off

#  # plot the angle between incident major axis and magnetic field
# figure(3)
# plot(elevs, [ray_O.init_pol_magfield_angle], 'b', 'linewidth', 2)
# hold on
# plot(elevs, [ray_O.init_wavedir_magfield_angle], 'b--', 'linewidth', 2)
# grid on
# min_y = min([[ray_O.init_pol_magfield_angle] ...
#             [ray_O.init_wavedir_magfield_angle]]) - 10  
# max_y = max([[ray_O.init_pol_magfield_angle] ...
#             [ray_O.init_wavedir_magfield_angle]]) + 10       
# min_y = floor(min_y/10)*10  
# min_y(min_y < 0) = 0  
# max_y = ceil(max_y/10)*10  

# set(gca, 'linewidth', 2, 'fontsize', 12, 'xlim', [0 90], 'ylim', [min_y max_y])
# xlabel('Transmit Elevation (degrees)', 'fontsize', 12)
# ylabel('Incident Angle (degrees)', 'fontsize', 12)
# title('Vertically polarized transmit antenna', 'fontsize', 12)
# hold off
# lh = legend('Pol. angle wrt {\bf \it B} field', ...
#             '{\bf \it k} direction wrt {\bf \it B} field')  