#
# Name :
#   NRT_ART_comparison.m
#
# Purpose :
#   Compares the spherical Earth 2D NRT and 3D (no magnetic field) NRT for a
#   fan of rays with ART. A spherically symmetric ionosphere is required with 
#   Quasi-Parabolic layers for the electron density profile. Increased
#   numerical errors with the NRT are noted when the ray apogee occurs at an 
#   altitude where the QP segment joins occur. This increased error is due to 
#   the nature of QP layers: while the height derivative of electron density
#   is matched at the joins, it is not possible to do so with the double 
#   derivative. See Figure 1.
#
# Calling sequence :
#   NRT_ART_comparison
#
# Inputs :
#   None
#
# Outputs :
#   None
#
# Change log:
#   V1.0  M.A. Cervera  03/03/2017
#     Initial Version
#
import math
from turtle import shape
import numpy as np  # py
import time
import ctypes as c
from Examples.ART.QP_profile_multi_seg import QP_profile_multi_seg

from Ionosphere import gen_iono_grid_3d as gen_iono
from pylap.raytrace_2d import raytrace_2d
from pylap.raytrace_3d import raytrace_3d
from Plotting import plot_ray_iono_slice
from Maths import raz2latlon
from Maths import latlon2raz
from Maths import ENU2xyz

#import raytrace_2d as raytrace
import matplotlib.pyplot as plt
#
# setup general stuff
# 
UT = [2000, 9, 21, 0, 0]                 # UT - year, month, day, hour, minute
speed_of_light = 2.99792458e8 
# plas_fac =  80.6163849431291d-6 	     
R12 = 100                            # R12 index
elevs = np.arange(3,81, .05, dtype = float)                  # initial elevation of rays
freqs = np.ones(len(elevs))*15         # frequency (MHz)
ray_bears = 30 + np.zeros(len(elevs))  # initial bearing of rays
re = 6376.138 

origin_lat = -20.0                   # latitude of the start point of rays
origin_long = 130.0                  # longitude of the start point of rays
origin_ht = 0.0                      # altitude of the start point of rays
doppler_flag = 1                     # interested in Doppler shift

tol = [1e-7, 0.01, 2]     # ODE solver tolerance and min/max allowable
                        # stepsizes. Try reducing (or increasing) the maximum 
                        # allowable stepsize and note the greatly improved
			# (or reduced)  numerical precision away from the
			# cusps and segment joins 
			
# close all

#
# generate 3D ionospheric, geomagnetic and irregularity grids
#
ht_start = 60           # start height for ionospheric grid (km)
ht_inc = 0.5            # height increment (km)
num_ht = 401            # number of  heights (must be < 401)
lat_start = -21.0 
lat_inc = 0.2 
num_lat = 201 
lon_start= 129.0 
lon_inc = 0.4 
num_lon = 45 

iono_grid_parms = [lat_start, lat_inc, num_lat, lon_start, lon_inc, num_lon, 
      ht_start, ht_inc, num_ht, ] 

B_ht_start = ht_start           # start height for geomagnetic grid (km)
B_ht_inc = 10                   # height increment (km)
B_num_ht = math.ceil(num_ht * ht_inc / B_ht_inc) + 0
B_lat_start = lat_start 
B_lat_inc = 1
B_num_lat = math.ceil(num_lat * lat_inc / B_lat_inc) +0
B_lon_start = lon_start 
B_lon_inc = 1
B_num_lon = math.ceil(num_lon * lon_inc / B_lon_inc)  +0
geomag_grid_parms = [B_lat_start, B_lat_inc, B_num_lat, B_lon_start, 
      B_lon_inc, B_num_lon, B_ht_start, B_ht_inc, B_num_ht] 


height_arr = np.arange(ht_start, ht_start + (num_ht)*ht_inc,ht_inc)
foE  = 3.0 
hmE  = 100.0 
ymE  = 25.0 
foF1 = 6.0 
hmF1 = 180.0 
ymF1 = 50.0 
foF2 = 10.0 
hmF2 = 250.0 
ymF2 = 75.0 
[en_profile, dN, QP_seg_coeffs] = QP_profile_multi_seg(foE, hmE, ymE, 
    foF1, hmF1, ymF1, foF2, hmF2, ymF2, height_arr, re) 

# "condition the bottom of the ionosphere" : Quasi-Parabolic layers are 
# discontinuous in the second derivative at the sgement joins. This can cause
# problems for the ODE solver of the NRT. The following conditioning can help
# reduce the numerical noise. Try the raytracing with and without. 
idx = min(np.argwhere(en_profile != 0)) 
en_profile[idx-1] = en_profile[idx]/8 
en_profile[idx-2] = en_profile[idx]/64 

# not interested in calculating Doppler shift
en_profile_5 = en_profile 

# not interested in calculating absorption so set collision frequency
# profile to zero
height_arr = np.arange(ht_start,  ht_start + (num_ht)*ht_inc,ht_inc)
cf_profile = np.zeros(len(height_arr)) 

iono_en_grid = np.zeros((num_lat,num_lon, num_ht)) 
iono_en_grid_5 = np.zeros((num_lat, num_lon, num_ht)) 
collision_freq = np.zeros((num_lat, num_lon, num_ht)) 

for ii in range (0, num_lat):
  for jj in range (0, num_lon):
    iono_en_grid[ii][jj] = en_profile 
    iono_en_grid_5 [ii][jj] = en_profile_5 
    collision_freq [ii][jj] = cf_profile 
  


# set B field to zero as we are only doing 'no-field' 3D NRT
Bx = np.zeros((B_num_lat, B_num_lon, B_num_ht))+2e-5 
By = np.zeros((B_num_lat, B_num_lon, B_num_ht))+2e-6 
Bz = np.zeros((B_num_lat, B_num_lon, B_num_ht))-5e-5 


#
# plot the electron density profile and its height derivative. Note where the
# discontinuities in the second derivative of electron density occurs.
#
# figure(1)
# fig1_pos = get(gcf, 'position') 
# fig1_pos(1) = 0 
# set(gcf, 'position', fig1_pos) 

fig, (ax1, ax2) = plt.subplots(2, 1)
# subplot(2,1,1)
# print(shape(QP_seg_coeffs))x
# print(shape(np.max(en_profile)))
ax1.scatter(height_arr, en_profile,  c= 'b', label = 'qp-segment-join',linewidth=0.001, linestyle=':', marker='.')
# print(en_profile[1])

# print en_profile
for i in range (0, 6):
      # print(QP_seg_coeffs[i, 3]-re)
      # print(en_profile[i])
      ax1.scatter(QP_seg_coeffs[i, 3]-re, en_profile[i]*1.05, c = 'r', linewidth = 0.3, linestyle =':', marker = '.')


# # plot(height_arr, en_profile, 'b', 'linewidth', 1)
# # # hold on
# # seg_h = plot([QP_seg_coeffs(:,3)-re QP_seg_coeffs(:,3)-re], 
# #              [0  max(en_profile)*1.05], 'r--', 'linewidth', 1) 
# # set(gca, 'Ylim', [0 max(en_profile)*1.05])
# # legend(seg_h(1), 'QP segment join')
# # ylabel('Electron Density (cm^-3)')
# # hold off
# # grid on



ax2.scatter(height_arr, dN, c = "b", label ='qp-segment-join', linewidth = .001, linestyle = ':', marker = '.')

# subplot(2,1,2)
# plot(height_arr, dN, 'b', 'linewidth', 1)
# # grid on  hold on
# ymin = -4e4 
# ymax = 2.5e4 
# seg_h = plot([QP_seg_coeffs(:,3)-re QP_seg_coeffs(:,3)-re], [ymin ymax], ...
#              'r--', 'linewidth', 1) 
# # legend(seg_h(1), 'QP segment join')	 
# # set(gca, 'Ylim', [ymin ymax])
# # hold off
# # xlabel('Height (km)')
# # ylabel('de_N/dh')


#
# call no-field 3D raytrace
# 

# nhops = 1                  # number of hops
# num_elevs = int(len(elevs)) 
# OX_mode = 0 

# print('3D NRT: generating #d ''no-field'' rays ...\n', num_elevs) 
# # tic

# ray_data_N = raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears, 
#               freqs, OX_mode, nhops, tol, iono_en_grid, 
# 	      iono_en_grid_5, collision_freq, iono_grid_parms, Bx, By, Bz, 
# 	      geomag_grid_parms) 

# # print (ray_data_N)
# print('num eleves :',num_elevs)
# idx_goodray = []
# for i in range(0, num_elevs):
#     if ray_data_N[i]['ray_label'] == 1:
#         idx_goodray.append(np.argwhere(ray_data_N[i]['ray_label'] == 1))


# group_range = []
# ground_range = []
# for i in range(0, num_elevs):
#     group_range.append(ray_data_N[i]['group_path'])
#     #ground_range.append(ray_data_N[i]['ground_range'])


# finished ray tracing with this ionosphere so clear it out of memory
# clear raytrace_3d_sp
# plot 3D NRT ground range vs elevation
# figure(2)
# fig2_pos = fig1_pos 
# fig2_pos(1) = fig1_pos(1) + fig1_pos(3)*1.02 
# set(gcf, 'position', fig2_pos) 
# fig = plt.figure()
# ax = fig.add_subplot(111)


# ax.scatter(elevs[0], ground_range[0], c='b',
#            marker='.', label='3D NRT range and elevaton')
# for i in range(0, len(idx_goodray)):
#       ax.scatter(elevs[i], ground_range[i], c='b',
#            marker='.')
# # plot(elevs(idx_goodray), ground_range(idx_goodray), 'b.', 'markersize', 8)
# hold on
# set(gca, 'fontsize', 14,'xtick', [0:5:40])
# grid on
# ray_apogee_3DNRT = np.ones(len(elevs))*np.NaN 
# ray_apogee_3DNRT(idx_goodray) = [ray_data_N(idx_goodray).apogee] 


# #
# # Now do 2D NRT
# #
# num_range = 600        
# range_inc = 5 
# max_range = range_inc * (num_range - 1) 
# iono_en_grid_2D = np.zeros(num_ht, num_range) 
# iono_en_grid_5_2D = np.zeros(num_ht, num_range) 
# collision_freq_2D = np.zeros(num_ht, num_range) 
# irreg = np.zeros(4, num_range) 
# for ii in range (num_range, 1):
#   iono_en_grid_2D[:, ii] = en_profile 
#   iono_en_grid_5_2D[:, ii] = en_profile_5 
#   collision_freq_2D[:, ii] = cf_profile 


# irregs_flag = 0 
# print('2D NRT: generating #d rays            ...\n', num_elevs) 
# # tic
# [ray_data, ray_path_data, ray_sv] = 
#     raytrace_2d_sp(elevs, ray_bears(1), freqs, nhops, tol, re, 
#                    irregs_flag, iono_en_grid_2D, iono_en_grid_5_2D, 
# 		   collision_freq_2D, ht_start, ht_inc, 
# 		   range_inc, irreg) 
# # NRT_total_time = toc 
# # fprintf('  NRT-only execution time = #f, Total mex execution time = #f\n\n', ...
# #         [ray_data.NRT_elapsed_time], NRT_total_time)

# idx_goodray_2D = np.argwhere([ray_data().ray_label] == 1) 
# group_range_2D = [ray_data().group_range] 
# ground_range_2D = [ray_data().ground_range] 


# # plot 2D NRT ground range vs elevation
# plot(elevs(idx_goodray_2D), ground_range_2D(idx_goodray_2D), 'r.', 
#     'markersize', 6)
# # ylabel('ground range (km)', 'fontsize', 14)
# # xlabel('elevation (degrees)', 'fontsize', 14)


# #
# # Now do ART
# #
# ground_ART = np.NaN.*zeros(size(elevs)) 
# group_ART = np.NaN.*zeros(size(elevs)) 
# phase_ART = np.NaN.*zeros(size(elevs)) 
# for ii in range(num_elevs, 1):
#   [gnd_ART, grp_ART, phas_ART] = ART_QP(elevs(ii), freqs(ii), re, QP_seg_coeffs) 
#   ground_ART(ii) = gnd_ART 
#   group_ART(ii) = grp_ART 
#   phase_ART(ii) = phas_ART 


# # plot ART ground range vs elevation
# # hold on
# plot(elevs, ground_ART, '.g', 'markersize', 4)
# # lh = legend(gca, '3D NRT', '2D NRT', 'ART') 
# # set(lh, 'fontsize', 14)
# # hold off


# #
# # Plot ray apogee of the 3D NRT rays vs elevation and indicate altitudes
# # where the second derivative of electron density wrt height is discontinuous
# #
# # figure(3)
# # fig3_pos = fig1_pos 
# # fig3_pos(1) = fig1_pos(1) + fig1_pos(3)*2.04 
# # set(gcf, 'position', fig3_pos) 
# plot(elevs, ray_apogee_3DNRT, 'b', 'linewidth', 1)
# # grid on
# # xlabel('Elevation (degrees)')
# # ylabel('Altitude (km)')
# # set(gca, 'fontsize', 14, 'xtick', [0:5:40], 'ytick', [60:20:260])
# # title('3D NRT ray apogee as a function of elevation', 'fontsize', 12)
# # hold on
# seg_h = plot([0 40], [QP_seg_coeffs(:,3)-re QP_seg_coeffs(:,3)-re], 
#              'r--', 'linewidth', 1) 
# # legend(seg_h(1), 'QP segment join', 'location', 'NorthWest')	 
# # hold off



# #
# # Plot difference between ART and 3D NRT vs elevation
# #
# # figure(4)
# # fig4_pos = fig1_pos 
# # fig4_pos(2) = fig1_pos(2) - fig1_pos(4)*1.02 - 100 
# # set(gcf, 'position', fig4_pos) 
# idx = np.argwhere(isnan(ground_ART) & [ray_data_N().ray_label] == 1)   
# #group_diff = [ray_data_N(:).group_range] - group_ART 
# #ground_diff = [ray_data_N(:).ground_range] - ground_ART 
# group_diff = group_range - group_ART 
# ground_diff = ground_range - ground_ART 
# plot(elevs(idx), ground_diff(idx) * 1000, 'b.')
# # hold on
# plot(elevs(idx), group_diff(idx) * 1000, 'r.')
# # hold off
# # set(gca, 'ylim', [-500 500], 'fontsize', 14, 'xtick', [0:5:40], 
# #          'ytick', [-500 :100 : 500])
# # ylabel('3D NRT - ART range (m)', 'fontsize', 14)
# # xlabel('elevation (degrees)', 'fontsize', 14)
# # lh = legend(gca, 'ground range', 'group range') 
# # set(lh, 'fontsize', 10)
# # grid on
# a3 = annotation('textarrow', [0.3411 0.2588], [0.7667 0.6961], 'String', 
#                 'E-F1 cusp') 
# a4 = annotation('textarrow', [0.3689 0.4393], [0.3469 0.4643], 'String', 
#                 'QP joining segment') 
# a5 = text(0.7106, -304.8752, 'N_e^\prime^\prime(h) discontinuity')  
# a6 = annotation('textarrow', [0.4484 0.5000], [0.2134 0.3119], 'String','F1-F2 cusp') 
# a7 = annotation('textarrow', [0.6786 0.6325], [0.3476 0.4612], 'String', 
#            'QP joining segment') 
# a8 = annotation('textarrow', [0.8089 0.8696], [0.6833 0.6333], 'String','F2 cusp') 
# a9 = text(22.7667, -304.7673, 'N_e^\prime^\prime(h) discontinuity')  
# a10 = text(0.75, 430, {'Note increased error at the cusps and',
#                       'where  N_e^\prime^\prime(h) is discontinuous'}) 

# # title(['log_{10}(tol) = ' num2str(log10(tol(1))) ', min/max step = ' 
# #        num2str(tol(2)) ' / ' num2str(tol(3)) 'km,  t = 
# #        num2str(time3, '#.1f') 's'], 'fontsize', 12)

# #
# # Plot difference between ART and 2D NRT vs elevation
# #
# # figure(5)
# # fig5_pos = fig1_pos 
# # fig5_pos(1) = fig1_pos(1) + fig1_pos(3)*1.02 
# # fig5_pos(2) = fig1_pos(2) - fig1_pos(4)*1.02 - 100 
# # set(gcf, 'position', fig5_pos) 
# idx = np.argwhere(~isnan(ground_ART) & [ray_data().ray_label] == 1)   
# group_diff = [ray_data().group_range] - group_ART 
# ground_diff = [ray_data().ground_range] - ground_ART 
# plot(elevs(idx), ground_diff(idx) * 1000, 'b.')
# # hold on
# plot(elevs(idx), group_diff(idx) * 1000, 'r.')
# # hold off
# # set(gca, 'ylim', [-500 500], 'fontsize', 14, 'xtick', [0:5:40], 
# #          'ytick', [-500 :100 : 500])
# # ylabel('2D NRT - ART range (m)', 'fontsize', 14)
# # xlabel('elevation (degrees)', 'fontsize', 14)
# # lh = legend(gca, 'ground range', 'group range') 
# # set(lh, 'fontsize', 10)
# # grid on
# txt_handle = text(0.75, 430, {'Note increased error at the cusps and ', 
#     'where N_e^\prime^\prime(h) is discontinuous'}) 


# #
# # Plot difference between 3D NRT and 2D NRT vs elevation
# #
# # figure(6)
# # fig6_pos = fig1_pos 
# # fig6_pos(1) = fig1_pos(1) + fig1_pos(3)*2.04 
# # fig6_pos(2) = fig1_pos(2) - fig1_pos(4)*1.02 - 100 
# # set(gcf, 'position', fig6_pos) 
# idx = np.argwhere([ray_data_N().ray_label] == 1 & [ray_data().ray_label] == 1)   
# group_diff = [ray_data_N().group_range] - [ray_data().group_range] 
# ground_diff = [ray_data_N().ground_range] - [ray_data().ground_range] 
# plot(elevs(idx), ground_diff(idx) * 1000, 'b.')
# # hold on
# plot(elevs(idx), group_diff(idx) * 1000, 'r.')
# # hold off
# # set(gca, 'ylim', [-500 500], 'fontsize', 14, 'xtick', [0:5:40], 
# #          'ytick', [-500 :100 : 500])
# # ylabel('3D NRT - 2D NRT range (m)', 'fontsize', 14)
# # xlabel('elevation (degrees)', 'fontsize', 14)
# # lh = legend(gca, 'ground range', 'group range') 
# # set(lh, 'fontsize', 10)
# # grid on

# txt_handle = text(0.75, 410, {'Note the error where N_e^\prime^\prime(h) is ' 
# 	'discontinuous is similar for 3D ' 'and 2D NRT'}) 
plt.show()