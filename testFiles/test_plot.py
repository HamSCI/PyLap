#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

tests the plot_ray_iono_slice function
assumes a file on the desktop which is a matlab mat file with all the variables
for the plot routnine
Created on Fri Jun 12 17:01:59 2020

@author: williamliles
"""
from plot_ray_iono_slice import plot_ray_iono_slice

def test_plot():
    import scipy.io as sio
    import numpy as np
    import matplotlib.pyplot as plt
    import plot_test
    UT = [2001, 3, 15, 7, 0]         #M UT - year, month, day, hour, minute
    R12 = 100
    ray_bear = 324.7             #M bearing of rays
    origin_lat = -23.5           #M latitude of the start point of ray
    origin_long = 133.7          #M longitude of the start point of ray
    freq = 15.0                  #M ray frequency (MHz)
    data = sio.loadmat('/home/william/Desktop/phiono.mat')
    # dict_keys(['__header__', '__version__', '__globals__', 'end_ht',
    # 'end_range', 'height_inc', 'iono_pf_subgrid', 'range_inc',
    # 'ray_path_data', 'start_ht', 'start_range'])
    end_ht = data.get('end_ht')[0][0]
    end_range = data.get('end_range')[0][0]
    height_inc = data.get('height_inc')[0][0]
    iono_grid = data.get('iono_pf_subgrid')
    range_inc = data.get('range_inc')[0][0]
    ray_path_data = data.get('ray_path_data')
    start_ht = data.get('start_ht')[0][0]
    start_range = data.get('start_range')[0][0]
    # all variables are now fine except for ray_path_data
    # turn it into a list of dictionaries
    # the keys are
    # initial_elev frequency ground_range height group_range phase_path
    # geometric_distance electron_density refractive_index collision_frequency
    # absorption
    
    
    ray_keys = ['initial_elev', 'frequency', 'ground_range', 'height',
                'group_range', 'phase_path', 'geometric_distance',
                'electron_density', 'refractive_index', 'collision_frequency',
                'absorption']
    ray_size = ray_path_data.shape[1]
    rays = [{}] * ray_size
    # print(ray_size)
    
    for ii in range(0, ray_size):
        rays[ii] = {'height': np.array([])}  # set up rays[] as type dict
        for jj in ray_keys:
            # rays[ii][jj] = np.empty(ray_path_data[0][ii][jj][0].size,dtype=
            #     float)
            # rays[ii][jj].append(ray_path_data[0][ii][jj][0])
            rays[ii][jj] = np.array([])
            rays[ii][jj] = ray_path_data[0][ii][jj][0]
    
    #
    # for jj in ray_keys:
        # print(jj)
        # print(type(rays[1][jj]))
        
    mat = {'iono_grid' : iono_grid, 'start_range': start_range, 'end_range':
           end_range, 'range_inc':range_inc, 'start_ht':start_ht, 'end_ht':
               end_ht, 'height_inc':height_inc, 'rays':rays}
    sio.savemat('/home/william/Desktop/slice.mat',mat)
    
 
    ax, ray_handle = plot_ray_iono_slice(iono_grid, start_range,
                      end_range, range_inc, start_ht, end_ht, height_inc,
                      rays,linewidth=1.5, color=[1, 1, 0.99])
#     UT_str = [num2str(UT(3)) '/' num2str(UT(2)) '/' num2str(UT(1)) '  ' ...
#           num2str(UT(4), '#M2.2d') ':' num2str(UT(5), '#M2.2d') 'UT']
# freq_str = [num2str(freq) 'MHz']
# R12_str = num2str(R12)
# lat_str = num2str(origin_lat)
# lon_str = num2str(origin_long)
# bearing_str = num2str(ray_bear)
# fig_str = [UT_str '   ' freq_str '   R12 = ' R12_str '   lat = ' lat_str ...
  #          ', lon = ' lon_str ', bearing = ' bearing_str]
# set(gcf, 'name', fig_str)
    fig_str_a = '{}/{}/{}  {:02d}:{:02d}UT   {}MHz   R12 = {}'.format(
              UT[2], UT[1], UT[0], UT[3], UT[4], freq, R12)
    fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
             origin_lat, origin_long, ray_bear)
    print(fig_str_a)
    print(fig_str_b)
    fig_str = fig_str_a + fig_str_b

    ax.set_title(fig_str)
    plt.show()
    
#M
#M Example 2 - Fan of rays, 3 hops, 30 MHz
#M
    data2 = sio.loadmat('/home/william/Desktop/phiono2.mat')
    # dict_keys(['__header__', '__version__', '__globals__', 'end_ht',
    # 'end_range', 'height_inc', 'iono_pf_subgrid', 'range_inc',
    # 'ray_path_data', 'start_ht', 'start_range'])
    end_ht2 = data2.get('end_ht')[0][0]
    end_range2 = data2.get('end_range')[0][0]
    height_inc2 = data2.get('height_inc')[0][0]
    iono_grid2 = data2.get('iono_pf_subgrid')
    range_inc2 = data2.get('range_inc')[0][0]
    ray_path_data2 = data2.get('ray_path_data')
    start_ht2 = data2.get('start_ht')[0][0]
    start_range2 = data2.get('start_range')[0][0]
    # all variables are now fine except for ray_path_data
    # turn it into a list of dictionaries
    # the keys are
    # initial_elev frequency ground_range height group_range phase_path
    # geometric_distance electron_density refractive_index collision_frequency
    # absorption
    
    
    ray_keys = ['initial_elev', 'frequency', 'ground_range', 'height',
                'group_range', 'phase_path', 'geometric_distance',
                'electron_density', 'refractive_index', 'collision_frequency',
                'absorption']
    ray_size2 = ray_path_data2.shape[1]
    rays2 = [{}] * ray_size
    # print(ray_size)
    
    for ii in range(0, ray_size2):
        rays2[ii] = {'height': np.array([])}  # set up rays[] as type dict
        for jj in ray_keys:
            # rays[ii][jj] = np.empty(ray_path_data[0][ii][jj][0].size,dtype=
            #     float)
            # rays[ii][jj].append(ray_path_data[0][ii][jj][0])
            rays2[ii][jj] = np.array([])
            rays2[ii][jj] = ray_path_data2[0][ii][jj][0]
    
    #



    
    #M plot the rays
    
    
    ax2, ray_handle2 = plot_ray_iono_slice(iono_grid2, start_range2,
                      end_range2, range_inc2, start_ht2, end_ht2, height_inc2,
                      rays2,linewidth=1.5, color='w')
    #freq_str = [num2str(freqs(1)) 'MHz']
    #fig_str = [UT_str '   ' freq_str '   R12 = ' R12_str '   lat = ' lat_str ...
    #       ', lon = ' lon_str ', bearing = ' bearing_str]
    #set(gcf, 'name', fig_str)
    freq = 30
    fig_str_a = '{}/{}/{}  {:02d}:{:02d}UT   {}MHz   R12 = {}'.format(
              UT[2], UT[1], UT[0], UT[3], UT[4], freq, R12)
    fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
             origin_lat, origin_long, ray_bear)
    print(fig_str_a)
    print(fig_str_b)
    fig_str = fig_str_a + fig_str_b
    ax2.set_title(fig_str)
    
    #set(gcf,'units','normal')
    #pos = get(gcf,'position')
    #pos(1) = 0.03
    #pos(2) = 0.28
    #set(gcf,'position', pos)
    
    #M plot three rays only
    #figure(3)
    data3 = sio.loadmat('/home/william/Desktop/phiono3.mat')
    #set(gcf, 'name', fig_str)    data2 = sio.loadmat('/home/william/Desktop/phiono2.mat')
    # dict_keys(['__header__', '__version__', '__globals__', 'end_ht',
    # 'end_range', 'height_inc', 'iono_pf_subgrid', 'range_inc',
    # 'ray_path_data', 'start_ht', 'start_range'])
    end_ht3 = data3.get('end_ht')[0][0]
    end_range3 = data3.get('end_range')[0][0]
    height_inc3 = data3.get('height_inc')[0][0]
    iono_grid3 = data3.get('iono_pf_subgrid')
    range_inc3 = data3.get('range_inc')[0][0]
    ray_path_data3 = data3.get('ray_path_data')
    start_ht3 = data3.get('start_ht')[0][0]
    start_range3 = data3.get('start_range')[0][0]
    # all variables are now fine except for ray_path_data
    # turn it into a list of dictionaries
    # the keys are
    # initial_elev frequency ground_range height group_range phase_path
    # geometric_distance electron_density refractive_index collision_frequency
    # absorption
    
    
    ray_keys = ['initial_elev', 'frequency', 'ground_range', 'height',
                'group_range', 'phase_path', 'geometric_distance',
                'electron_density', 'refractive_index', 'collision_frequency',
                'absorption']
    ray_size3 = ray_path_data3.shape[1]

    rays3 = [{}] * ray_size
    # print(ray_size)


    for ii in range(0, ray_size3):
        rays3[ii] = {'height': np.array([])}  # set up rays[] as type dict
        for jj in ray_keys:
            # rays[ii][jj] = np.empty(ray_path_data[0][ii][jj][0].size,dtype=
            #     float)
            # rays[ii][jj].append(ray_path_data[0][ii][jj][0])
            rays3[ii][jj] = np.array([])
            rays3[ii][jj] = ray_path_data3[0][ii][jj][0]
    
    #
    # for jj in ray_keys:
        # print(jj)
        # print(type(rays[1][jj]))


    
    #M plot the rays
    
    
    ax3, ray_handle3 = plot_ray_iono_slice(iono_grid3, start_range3,
                      end_range3, range_inc3, start_ht3, end_ht3, height_inc3,
                      rays3[0:3],linewidth=1.5, color='w')
    

    #[axis_handle, ray_handle] = plot_ray_iono_slice(iono_pf_subgrid, ...
    #  start_range, end_range, range_inc, start_ht, end_ht, height_inc, ...
    # ray_path_data(1:3), 'color', 'w', 'linewidth', 2)
    #set(ray_handle(1), 'linestyle', '--')
    #set(ray_handle(2), 'linestyle', ':')
    ray_handle3[0][0].set_linestyle('--')
    ray_handle3[1][0].set_linestyle(':')
    ax3.set_title(fig_str)
    
    #set(gcf,'units','normal')
    #pos = get(gcf,'position')
    #pos(1) = 0.05
    #pos(2) = 0.12
    #set(gcf,'position', pos)
    
    #fprintf('\n')

    plt.show()

test_plot()
    