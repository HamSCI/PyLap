#
# Name :
#   NRT_comparison.m
#
# Purpose :
#   Compares the WGS84 2D NRT and 3D (no magnetic field) NRT for a fan of rays
#   in an ionosphere with a down range gradient (no cross-range gradients).
#
# Calling sequence :
#   NRT_comparison
#
# Inputs :
#   None
#
# Outputs :
#   None
#
# Notes :
#  The ionospheric grid is constructed from QP layers. The foF2 is modified
#  down-range to create the down-range electron density gradient. The QP
#  ionosphere allows us to ensure the ionospheric grid is smooth. Note how
#  well the 2D and 3D no-field NRT engines agree. You can uncomment code
#  blocks below to use IRI instead of QP layers. The agreement is not so good,
#  presumably this due to smoothness issues with IRI.
#
# Change log:
#   V1.0  M.A. Cervera  11/07/2017
#     Initial Version
#
import math
from turtle import shape
import numpy as np  # py
import time
import ctypes as c
from Examples.ART.QP_profile_multi_seg import QP_profile_multi_seg
#import plot_ray_iono_slice as plot_iono


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
UT = [2000, 9, 21, 0, 0]           # UT - year, month, day, hour, minute
speed_of_light = 2.99792458e8
R12 = 100
# initial elevation of rays
elevs = np.arange(3, 81.05, 0.05, dtype=float)
freqs = np.ones(len(elevs))*15      # frequency (MHz)
ray_bears = np.zeros(len(elevs))    # initial bearing of rays - due north
# ODE solver tolerance and min max stepsizes
tol = [1e-8, 0.01, 2]
origin_lat = -20.0                # latitude of the start point of rays
origin_long = 130.0               # longitude of the start point of rays
origin_ht = 0.0                   # altitude of the start point of rays
doppler_flag = 1                  # interested in Doppler shift
re = 6376.0                       # radius of Earth - only required for
# ionospheric QP layer generation

# close all

print('\n' 'Comparison of 3D NRT ("no-field") and 2D NRT for a WGS84 Earth.\n')
print('The ionosphere used has down-range gradients but NO cross-range \n')
print('gradients.\n\n')


#
# generate 3D ionospheric, geomagnetic and irregularity grids
#
ht_start = 60          # start height for ionospheric grid (km)
ht_inc = 1             # height increment (km)
num_ht = 400           # number of  heights (must be < 201)
lat_start = -21.0
lat_inc = 0.2
num_lat = 201
lon_start = 129.0
lon_inc = 0.5
num_lon = 6

iono_grid_parms = [lat_start, lat_inc, num_lat, lon_start, lon_inc, num_lon,
                   ht_start, ht_inc, num_ht, ]

B_ht_start = ht_start          # start height for geomagnetic grid (km)
B_ht_inc = 10                  # height increment (km)
B_num_ht = math.ceil(num_ht * ht_inc / B_ht_inc) + 1
B_lat_start = lat_start
B_lat_inc = 1.0
B_num_lat = math.ceil(num_lat * lat_inc / B_lat_inc) + 1
B_lon_start = lon_start
B_lon_inc = 1.0
B_num_lon = math.ceil(num_lon * lon_inc / B_lon_inc) + 1
geomag_grid_parms = [B_lat_start, B_lat_inc, B_num_lat, B_lon_start,
                     B_lon_inc, B_num_lon, B_ht_start, B_ht_inc, B_num_ht]

iono_en_grid = np.zeros((num_lat, num_lon, num_ht))
iono_en_grid_5 = np.zeros((num_lat, num_lon, num_ht))
collision_freq = np.zeros((num_lat, num_lon, num_ht))

#
# The following commented out code generates an IRI gridded ionosphere
#
# for ii = 1:num_lat
#   # generate ionospheric electron density profile
#   lat = lat_start + (ii-1)*lat_inc
#   lon = origin_long
#   [iono, iono_extra] = iri2016(lat, lon, R12, UT, ht_start, ht_inc, num_ht)
#   en_profile = iono(1, :) / 1e6      # convert to electrons per cm^3
#   idx = find(en_profile < 0)
#   en_profile(idx) = 0
#   en_profile_5 = en_profile          # not interested in Doppler shift
#
#   # not interested in calculating absorption so set collision frequency
#   # profile to zero
#   cf_profile = zeros(size(height_arr))
#
#   for jj = 1:num_lon
#     iono_en_grid(ii, jj, :) = en_profile
#     iono_en_grid_5(ii, jj, :) = en_profile_5
#     collision_freq(ii, jj, :) = cf_profile
#   end
# end

foE = 3.0
hmE = 100.0
ymE = 25.0
foF1 = 5.0
hmF1 = 180.0
ymF1 = 50.0
foF2_init = 10.0
hmF2 = 250.0
ymF2 = 75.0
foF2_increase_per_deglat = 5 / ((num_lat-1) * lat_inc)
height_arr = np.arange(ht_start, ht_start + (num_ht)*ht_inc, ht_inc)
for ii in range(0, num_lat):
    lat = lat_start + (ii)*lat_inc

    # apply N/S gradient to foF2
    foF2 = foF2_init + (lat-origin_lat)*foF2_increase_per_deglat

    # generate ionospheric electron density profile
    [en_profile, dN, QP_seg_coeffs] = QP_profile_multi_seg(foE, hmE, ymE, foF1, hmF1, ymF1,
                                                           foF2, hmF2, ymF2, height_arr, re)

    # "condition the bottom of the ionosphere" : Quasi-Parabolic layers are
    # discontinuous in the second derivative at the sgement joins. This can
    # cause problems for the ODE solver of the NRT. The following conditioning
    # can help reduce the numerical noise. Try the raytracing with and without.
    idx = min(np.argwhere(en_profile != 0))
    en_profile[idx-1] = en_profile[idx]/8
    en_profile[idx-2] = en_profile[idx]/64

    # not interested in Doppler shift
    en_profile_5 = en_profile

    # not interested in calculating absorption so set collision frequency
    # profile to zero
    cf_profile = np.zeros(len(height_arr))

    for jj in range(0, num_lon):
        iono_en_grid[ii][jj] = en_profile
        iono_en_grid_5[ii][jj] = en_profile_5
        collision_freq[ii][jj] = cf_profile

# set B field to zero as we are only doing 'no-field' 3D NRT
Bx = np.zeros((B_num_lat, B_num_lon, B_num_ht))
By = np.zeros((B_num_lat, B_num_lon, B_num_ht))
Bz = np.zeros((B_num_lat, B_num_lon, B_num_ht))


#
# call no-field 3D raytrace
#
nhops = 1                  # number of hops
num_elevs = len(elevs)
OX_mode = 0

print("3D NRT: generating", num_elevs, "\'no-field\' rays ...")
tic = time.time
[ray_data_N, ray_N, ray_sv_N] = \
    raytrace_3d(origin_lat, origin_long, origin_ht, elevs, ray_bears,
                freqs, OX_mode, nhops, tol, iono_en_grid,
                iono_en_grid_5, collision_freq, iono_grid_parms, Bx, By, Bz,
                geomag_grid_parms)
NRT_total_time = toc = time.time
# print('\n   NRT-only execution time = #f, Total mex execution time = #f\n\n',
#       [ray_data_N.NRT_elapsed_time], NRT_total_time)

idx_goodray = []
for i in range(0, num_elevs):
    if ray_data_N[i]['ray_label'] == 1:
        idx_goodray.append(np.argwhere(ray_data_N[i]['ray_label'] == 1))

group_range = []
ground_range = []
for i in range(0, num_elevs):
    group_range.append(ray_data_N[i]['group_range'])
    ground_range.append(ray_data_N[i]['ground_range'])


#
# generate the 2D electron density grid
#
num_range = 200
range_inc = 20
iono_en_grid_2D = np.zeros((num_ht, num_range))
iono_en_grid_5_2D = np.zeros((num_ht, num_range))
collision_freq_2D = np.zeros((num_ht, num_range))
irreg = np.zeros((4, num_range))

#
# The following commented out code generates an IRI gridded ionosphere
#
# for ii = 1:num_range
#
#   # generate iono profile
#   azim = ray_bears(1)
#   range = (ii - 1)*range_inc*1000
#   [lat, lon] = raz2latlon(range, azim, origin_lat, origin_long, 'wgs84')
#
#   [iono, iono_extra] = iri2016(lat, origin_long, R12, UT, ht_start, ht_inc, ...
#                                num_ht)
#   en_profile = iono(1, :) / 1e6      # convert to electrons per cm^3
#   idx = find(en_profile < 0)
#   en_profile(idx) = 0
#   en_profile_5 = en_profile          # not interested in Doppler shift
#
#
#   # not interested in calculating absorption so set collision frequency
#   # profile to zero
#   cf_profile = zeros(size(height_arr))
#
#   iono_en_grid_2D(:, ii) = en_profile
#   iono_en_grid_5_2D(:, ii) = en_profile_5
#   collision_freq_2D(:, ii) = cf_profile
# end

for ii in range(0, num_range):
    azim = [ray_bears[0]]
    ranges = (ii)*range_inc*1000
    [lat, lon] = raz2latlon.raz2latlon(
        ranges, azim, origin_lat, origin_long, 'wgs84')

    # apply N/S gradient to foF2
    foF2 = foF2_init + (lat - origin_lat)*foF2_increase_per_deglat

    # generate ionospheric electron density profile
    [en_profile, dN, QP_seg_coeffs] = QP_profile_multi_seg(foE, hmE, ymE, foF1, hmF1, ymF1,
                                                           foF2, hmF2, ymF2, height_arr, re)

    # "condition the bottom of the ionosphere" : Quasi-Parabolic layers are
    # discontinuous in the second derivative at the sgement joins. This can
    # cause problems for the ODE solver of the NRT. The following conditioning
    # can help reduce the numerical noise. Try the raytracing with and without.
    idx = min(np.argwhere(en_profile != 0))
    en_profile[idx-1] = en_profile[idx]/8
    en_profile[idx-2] = en_profile[idx]/64

    # not interested in Doppler shift
    en_profile_5 = en_profile

    # not interested in calculating absorption so set collision frequency
    # profile to zero
    cf_profile = np.zeros(len(height_arr))

    iono_en_grid_2D[:, ii] = en_profile
    iono_en_grid_5_2D[:, ii] = en_profile_5
    collision_freq_2D[:, ii] = cf_profile

#
# call the 2D raytrace engine
#
irregs_flag = 0
print('2D NRT: generating', num_elevs, '\'no-field\' rays ...')
tic
[ray_data, ray_path_data, ray_state_vec] = \
    raytrace_2d(origin_lat, origin_long, elevs, ray_bears[0], freqs, nhops,
                tol, irregs_flag, iono_en_grid_2D, iono_en_grid_5_2D,
                collision_freq_2D, ht_start, ht_inc, range_inc, irreg)
NRT_total_time = toc
# print('\n   NRT-only execution time = #f, Total mex execution time = #f\n\n',
#       [ray_data.NRT_elapsed_time], NRT_total_time)

idx_goodray_2D = []
for i in range(0, num_elevs):
    if ray_data[i]['ray_label'] == 1:
        idx_goodray_2D.append(np.argwhere(ray_data[i]['ray_label'] == 1))

group_range_2D = []
ground_range_2D = []
for i in range(0, num_elevs):
    group_range_2D.append(ray_data[i]['group_range'])
    ground_range_2D.append(ray_data[i]['ground_range'])


# finished ray tracing with this ionosphere so clear it out of memory
# clear raytrace_3d

#
# now for some plots
#

# plot the 3D and 2D results vs elevation
# figure(1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(elevs[0], ground_range[0], c='b',
           marker='.', label='3D NRT (no field)')
for i in range(0, len(idx_goodray)):
    ax.scatter(elevs[i], ground_range[i], c='b', marker='.')
# hold on
# set(gca, 'fontsize', 14)
# grid on
ax.scatter(elevs[0], ground_range_2D[0], c='r', marker='.', label='2D NRT')
for i in range(0, len(idx_goodray_2D)):
    ax.scatter(elevs[i], ground_range_2D[i], c='r', marker='.')
# set(gca, 'xlim', [0 45], 'xtick', [0:5:45])
ax.set_ylabel('ground range (km)', fontsize=14)
ax.set_xlabel('elevation (degrees)', fontsize=14)

ax.legend()
# plt.show()
# hold off

# fig1_pos = get(gcf, 'position')
# fig2_pos = fig1_pos
# fig1_pos[1] = fig1_pos[1] - 300
# set(gcf, 'position', fig1_pos)

# fig2_pos[1] = fig2_pos(1) + 300


# plot the difference between the 3D and 2D NRT results
# figure(2)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
# set(gcf, 'position', fig2_pos)
idx = []
for i in range(0, num_elevs):
    if ray_data_N[i]['group_range'] == 1:
        idx.append(i)
    if ray_data_N[i]['ray_label'] == 1:
        idx.append(i)

group_diff = np.zeros(num_elevs)
ground_diff = np.zeros(num_elevs)
for i in range(0, num_elevs):
    group_diff[i] = (ray_data_N[i]['group_range'] - ray_data[i]['group_range'])
    ground_diff[i] = (ray_data_N[i]['ground_range'] - ray_data[i]['ground_range'])


ax2.scatter(elevs[0], (ground_diff[0] * 1000),
            c='b', marker='.', label='ground range')
# hold on
ax2.scatter(elevs[0], (group_diff[0] * 1000),
            c='r', marker='.', label='group range')
for i in range(0, len(idx)):
    ax2.scatter(elevs[i], (ground_diff[i] * 1000), c='b', marker='.')
    ax2.scatter(elevs[i], (group_diff[i] * 1000), c='r', marker='.')
ax2.legend()
ax2.set_ylabel('3D NRT (no field) - 2D NRT range (m)')
ax2.set_xlabel('elevation (degrees)')
ax2.set_ylim(-500, 500)
plt.show()
# hold off
# set(gca, 'xlim', [0, 45], 'ylim', [-500, 500], 'fontsize', 14,
#          'xtick', np.arange(0, 45, 5, dtype=float), 'ytick', np.arange(-500,500,100, dtype=float))

# lh = legend(gca, 'ground range', 'group range')
# set(lh, 'fontsize', 14)
# grid on
