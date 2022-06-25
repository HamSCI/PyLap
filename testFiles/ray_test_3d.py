#M
#M Name :
#M   ray_test_3d.m
#M
#M Purpose :
#M   Example of using raytrace_3d for a fan of rays. 
#M
#M Calling sequence :
#M   ray_test_3d
#M
#M Inputs :
#M   None
#M
#M Outputs :
#M   None
#M
#M Modification History:
#M   V1.0  M.A. Cervera  07/12/2009
#M     Initial version.
#M
#M   V1.1  M.A. Cervera  12/05/2009
#M     Uses 'parfor' to parallelize the computation if the parallel computing 
#M     tool box is available
#M
#M   V1.3  M.A. Cervera  19/05/2011
#M     More efficient handling of ionospheric  and geomagnetic grids grids in
#M     call to raytrace_3d  
#M
#M   V2.0 M.A. Cervera  03/05/2016
#M     Modified to make use of multi-threaded raytrace_3d. IRI2016 is now used
#M     to generate the ionosphere.
#M
#    W. C. Liles 10/08/2020
#      Convert to Python
#
#!/usr/bin/python
import numpy as np

#M
#M setup general stuff
#M
UT = [2000 9 21 0 0]           #M UT - year, month, day, hour, minute
speed_of_light = 2.99792458e8
R12 = 100
elevs = np.arange(3,81,1)               #M initial elevation of rays
freqs = np.ones(elevs.size, dtype = float) * 15   #M frequency (MHz)
ray_bears = np.zeros(elevs.size, dtype = float) #M initial bearing of rays
origin_lat = -20.0             #M latitude of the start point of rays
origin_long = 130.0            #M longitude of the start point of rays
origin_ht = 0.0                #M altitude of the start point of rays
doppler_flag = 1               #M interested in Doppler shift
, dtype = float

print('\n' 
   'Example of 3D magneto-ionic numerical raytracing for a WGS84 ellipsoidal' 
   ' Earth\n\n')

#M
#M generate ionospheric, geomagnetic and irregularity grids
#M
ht_start = 60          #M start height for ionospheric grid (km)
ht_inc = 2             #M height increment (km)
num_ht = 201           
lat_start = -20.0
lat_inc = 0.3
num_lat = 101.0
lon_start= 128.0
lon_inc = 1.0
num_lon = 5.0
iono_grid_parms = [lat_start, lat_inc, num_lat, lon_start, lon_inc, num_lon, ...
      ht_start, ht_inc, num_ht, ]

B_ht_start = ht_start          #M start height for geomagnetic grid (km)
B_ht_inc = 10                  #M height increment (km)
B_num_ht = ceil(num_ht .* ht_inc ./ B_ht_inc)
B_lat_start = lat_start
B_lat_inc = 1.0
B_num_lat = ceil(num_lat .* lat_inc ./ B_lat_inc)
B_lon_start = lon_start
B_lon_inc = 1.0
B_num_lon = ceil(num_lon .* lon_inc ./ B_lon_inc) 
geomag_grid_parms = [B_lat_start, B_lat_inc, B_num_lat, B_lon_start, ...
      B_lon_inc, B_num_lon, B_ht_start, B_ht_inc, B_num_ht]


tic
fprintf('Generating ionospheric and geomag grids... ')
[iono_pf_grid, iono_pf_grid_5, collision_freq, Bx, By, Bz] = ...
    gen_iono_grid_3d(UT, R12, iono_grid_parms, ...
                     geomag_grid_parms, doppler_flag)
toc
fprintf('\n')

#M convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid = iono_pf_grid.^2 / 80.6164e-6
iono_en_grid_5 = iono_pf_grid_5.^2 / 80.6164e-6

#M
#M call raytrace
#M
nhops = 4                  #M number of hops
tol = [1e-7 0.01 25]       #M ODE solver tolerance and min max stepsizes
num_elevs = length(elevs)

#M Generate the O mode rays
OX_mode = 1
 
fprintf('Generating #Md O-mode rays ...', num_elevs)
tic
[ray_data_O, ray_O, ray_state_vec_O] = ...
  raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, freqs, ...
              OX_mode, nhops, tol, iono_en_grid, iono_en_grid_5, ...
	          collision_freq, iono_grid_parms, Bx, By, Bz, ...
	          geomag_grid_parms)
	      
NRT_total_time = toc
fprintf('\n   NRT-only execution time = #Mf, Total mex execution time = #Mf\n\n', ...
        [ray_data_O.NRT_elapsed_time], NRT_total_time)


for rayId=1:num_elevs
  num = length(ray_O(rayId).lat)
  ground_range = zeros(1, num)
  lat = ray_O(rayId).lat
  lon = ray_O(rayId).lon 
  ground_range(2:num) = latlon2raz(lat(2:num), lon(2:num), origin_lat, ...
      origin_long,'wgs84')/1000.0
  ray_O(rayId).ground_range = ground_range
end


#M Generate the X mode rays - note in the raytrace_3d call the ionosphere does
#M not need to be passed in again as it is already in memory
OX_mode = -1
 
fprintf('Generating #Md X-mode rays ...', num_elevs)
tic
[ray_data_X, ray_X, ray_sv_X] = ...
  raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, freqs, ...
              OX_mode, nhops, tol)
NRT_total_time = toc
fprintf('\n   NRT-only execution time = #Mf, Total mex execution time = #Mf\n\n', ...
        [ray_data_X.NRT_elapsed_time], NRT_total_time)

for rayId=1:num_elevs
  num = length(ray_X(rayId).lat)
  ground_range = zeros(1, num)
  lat = ray_X(rayId).lat
  lon = ray_X(rayId).lon    
  ground_range(2:num) = latlon2raz(lat(2:num), lon(2:num), origin_lat, ...
      origin_long,'wgs84')/1000.0
  ray_X(rayId).ground_range = ground_range
end


#M Generate the rays for the case where the magnetic field is ignored  - note
#M in the raytrace_3d call the ionosphere does not need to be passed in again
#M as it is already in memory
OX_mode = 0
 
fprintf('Generating #Md ''no-field'' rays ...', num_elevs)
tic
[ray_data_N, ray_N, ray_sv_N] = ...
  raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, freqs, ...
              OX_mode, nhops, tol)
NRT_total_time = toc
fprintf('\n   NRT-only execution time = #Mf, Total mex execution time = #Mf\n\n', ...
        [ray_data_N.NRT_elapsed_time], NRT_total_time)

for rayId=1:num_elevs
  num = length(ray_N(rayId).lat)
  ground_range = zeros(1, num)
  lat = ray_N(rayId).lat
  lon = ray_N(rayId).lon    
  ground_range(2:num) = latlon2raz(lat(2:num), lon(2:num), origin_lat, ...
      origin_long,'wgs84')/1000.0
  ray_N(rayId).ground_range = ground_range
end

fprintf('\n')


#M finished ray tracing with this ionosphere so clear it out of memory
clear raytrace_3d


#M plot the rays
figure(1)
pos = get(gcf, 'position')
pos(3) = pos(3)*1.5
pos(4) = pos(4)*1.5
set(gcf, 'position', pos)
plot3(ray_O(1).lat, mod(ray_O(1).lon, 360), ray_O(1).height, '.b', ...
      'markersize', 5)
set(gca, 'Zlim', [0 500])
hold on
plot3(ray_X(1).lat,  mod(ray_X(1).lon, 360), ray_X(1).height, '.r',  ...
      'markersize',5)
plot3(ray_N(1).lat,  mod(ray_N(1).lon, 360), ray_N(1).height, 'g')
for ii = 3:2:num_elevs
  plot3(ray_O(ii).lat, mod(ray_O(ii).lon, 360), ray_O(ii).height, '.b', ...
        'markersize', 5)
  plot3(ray_X(ii).lat, mod(ray_X(ii).lon, 360), ray_X(ii).height, '.r', ...
        'markersize', 5)
  plot3(ray_N(ii).lat,  mod(ray_N(ii).lon, 360), ray_N(ii).height, 'g')
end  
hold off
grid on
xlabel('latitude (deg)')
ylabel('longitude (deg)')
zlabel('Height (km)')
legend('O Mode', 'X Mode', 'No Mag-field')

figure(2)
start_range = 0
end_range = 2000
range_inc = 50
end_range_idx = int((end_range-start_range) / range_inc) + 1
start_ht = 0
start_ht_idx = 1
height_inc = 5
end_ht = 350
end_ht_idx = fix(end_ht / height_inc) + 1
iono_pf_subgrid = np.zeros(end_ht_idx, end_range_idx)
plot_ray_iono_slice(iono_pf_subgrid, start_range, end_range, range_inc, ...
    start_ht, end_ht, height_inc, ray_O, linewidth=1.5, color=[1, 1, 0.99])
