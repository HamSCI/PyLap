#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:41:22 2020

@author: william

"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from plot_ray_iono_slice import plot_ray_iono_slice
from Plot_2D_slice import Plot_2D_slice
import sys
def test_class():
    UT = [2001, 3, 15, 7, 0]         #M UT - year, month, day, hour, minute
    R12 = 100
    ray_bear = 324.7             #M bearing of rays
    origin_lat = -23.5           #M latitude of the start point of ray
    origin_long = 133.7          #M longitude of the start point of ray
    freq = 15.0       
    mat = sio.loadmat('/home/william/Desktop/slice.mat',squeeze_me=True)
    # mat2.keys()
    # dict_keys(['__header__', '__version__', '__globals__', 'iono_grid',
    #'start_range', 'end_range', 'range_inc', 'start_ht', 'end_ht',
    # 'height_inc', 'rays'])
    iono_grid = mat.get('iono_grid')
    start_range = mat.get('start_range')
    end_range = mat.get('end_range')
    range_inc = mat.get('range_inc')
    start_ht = mat.get('start_ht')
    end_ht = mat.get('end_ht')
    height_inc = mat.get('height_inc')
    rays = mat.get('rays')
    # converts rays from a numpy structure to a dictionary
    ray_list = []
    for ray_idx in rays:
        ray_t = {}
        names = ray_idx.dtype.names
        for name_idx in names:
            ray_t[name_idx] = np.array(ray_idx[name_idx].tolist())
        ray_list.append(ray_t)
    class_flag = True  # if true use object Plot_2D_slice, false use plot_ray
            
    if class_flag:
        plot2d = Plot_2D_slice()
        plot2d.set_boundries(start_range, end_range, range_inc,
                      start_ht, end_ht, height_inc)
        ax, fig, image = plot2d.set_iono_grid(iono_grid)
        plot2d.show_color_bar(ax, fig,image)
        ray_handle = plot2d.show_rays(ray_list,linewidth=1.5, 
                                      color=[1, 1, 0.99])
        h = plot2d.zenith_line(end_range/4, -50, end_ht + 100, linewidth=1.5, 
                        color='b')
        a,b,c,d,e =plot2d.get_im_coords(end_range*3/4)
        x = np.array([a,c])
        y = np.array([b,d])
        plt.plot(x,y,'k')
    else:
        ax, ray_handle = plot_ray_iono_slice(iono_grid, start_range,
                          end_range, range_inc, start_ht, end_ht, height_inc,
                          ray_list,linewidth=1.5, color=[1, 1, 0.99])
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
    
    print(' to end')
    plt.show()
test_class()
    