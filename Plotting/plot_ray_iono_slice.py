#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:49:47 2020

@author: williamliles
"""

#M
#M Name :
#M   plot_ray_iono_slice.m
#M
#M Purpose:
#M   Plots ionospheric slice in an arc (to preserve curved Earth geometry)
#M   and overplots rays
#M
#M Calling sequence:
#M   plot_ray_iono_slice(iono_grid, start_range, end_range, ...
#M           range_inc, start_height, end_height, height_inc, ray)
#M
#M   [axis_handle, ray_handle] = plot_ray_iono_slice(iono_grid, start_range,
#M           end_range, range_inc, start_height, end_height, height_inc, ray)
#M
#M Inputs:
#M   iono_grid    - grid (height, ground range) of ionospheric plasma
#M                    frequencies (MHz)
#M   start_range  - starting ground range of iono_grid (km)
#M   end_range    - final ground range of iono_grid (km)
#M   range_inc    - ground range increment of iono_grid (km)
#M   start_height - starting height of iono_grid (km)
#M   end_height   - final height of iono_grid (km)
#M   height_inc   - height increment of iono_grid (km)
#M   ray          - structure containing ray information:
#M     ray(N).ground_range  = ground range vector for Nth ray
#M     ray(N).height        = height vector for Nth ray
#M
#M     Notes:
#M     1. The ground range of the rays can also be input with ray(N).gndrng .
#M        However, if both of the gndrng and ground_range fields are defined,
#M        then the latter will be used.
#M     2. If rays are not required to be plotted, set ray = []
#M
#M Optional Inputs:
#M   The required inputs can be followed by parameter/value pairs to specify
#M   additional properties of the rays. These parameter/value pairs are any
#M   that are accepted by the MATLAB inbuilt plot function. For example,
#M
#M   plot_ray_iono_slice(iono_pf_subgrid, start_range, end_range, range_inc, ...
#M        start_ht, end_ht, height_inc, ray(8:9), 'color', 'w', 'linewidth', 2)
#M
#M   will create white rays with line width of 2 points.
#M
#M Outputs:
#M   axis_handle  - the handle of the plot axis
#M   ray_handle   - vector containing handle for each ray
#M
#M Modification History:
#M   14/05/2008  V1.0  M. A. Cervera
#M     Initial Version.
#M
#M   24/06/2009  V1.1  M. A. Cervera
#M     Minor modification to font sizes to improve display on small screens
#M
#M   06/07/2015  V1.2  M. A. Cervera
#M      Minor modification to limit the size of the displayed figure window on
#M      large screens when using software opengl. This is to mitigate framebuffer
#M      issues.
#M

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import platform
from qtpy.QtWidgets import QApplication

# function [axis_handle, ray_handle] = plot_ray_iono_slice(iono_grid, ...
#      start_range, end_range, range_inc, start_height, end_height, ...
#      height_inc, ray, varargin)
def plot_ray_iono_slice(iono_grid, start_range, end_range, range_inc,
                start_height, end_height, height_inc, ray, **kwargs):
    #
    #
    #
    matplotlib.use('TkAgg')
    #
    #   
    #M
    #M Input consistency and error checking
    #M
    #iono_grid_size = size(iono_grid)
    iono_grid_size = iono_grid.shape
    # print(type(ray[0]['ground_range']))
    # print(ray[0]['ground_range'])
    # print(iono_grid_size.shape)
    
    #if length([start_height : height_inc : end_height]) ~= iono_grid_size(1)
    #  error('start_height, end_height and height_inc inconsistent with iono_grid')
    #  return
    #end
    #if length([start_range : range_inc : end_range]) ~= iono_grid_size(2)
    #  error('start_range, end_range and range_inc inconsistent with iono_grid')
    #  return
    #end
    #
    #if ~isempty(ray)
    #  if (~isfield(ray, 'height')) | ...
    #     (~isfield(ray, 'gndrng') & ~isfield(ray, 'ground_range'))
    #    error('input ray is not correct type')
    #    return
    #  end
    #end
    #
    # heights = np.linspace(start_height, end_height, int((end_height-start_height) / height_inc),True,False,int)
    # no_heights = heights.size + 1
    # # print(start_height,end_height, height_inc, no_heights)
    # # print(heights)
    # if no_heights != iono_grid_size[0]:
    #         # import ipdb; ipdb.set_trace()
    #         print('start_height, end_height and height_inc inconsistent '
    #               'with iono_grid in plot_ray_iono_slice')
    #         sys.exit()
    # ranges = np.linspace(start_range, end_range, int(((end_range - start_range)
    #                      + 1) / range_inc))
    # no_ranges = ranges.size + 1
    # if no_ranges != iono_grid_size[1]:
    #         print('start_range, end_range and range_inc inconsistent with '
    #               'iono_grid in plot_ray_iono_slice')
    #         sys.exit()


    #if isfield(ray, 'ground_range')
    #  for rayId=1:length(ray)
    #    ray(rayId).gndrng = ray(rayId).ground_range
    #  end
    #end
    #
    #for idx=1:length(ray)
    #  if length(ray(idx).height) ~= length(ray(idx).gndrng)
    #    error('ray height and ground range vectors have diffent lengths')
    #    return
    #  end
    #end
    # print(len(ray))
    for rayID in range(0, len(ray)):
        if 'ground_range' in ray[rayID].keys():
          ray[rayID]['gndrng'] = ray[rayID]['ground_range']

    # import ipdb; ipdb.set_trace()
    for ii in range(len(ray)):
      # print(ii)
      if(~np.isnan(np.all(ray[ii]['gndrng']))):
        np.nan_to_num(ray[ii]['gndrng'],False,0.0)
      # print(ii, (ray[ii]['height']), (ray[ii]['gndrng']))
      if len(ray[ii]['height']) != len(ray[ii]['gndrng']):
        print('ray height and ground range vectors have diffent lengths' +
                  ' in plot_ray_iono_slice')
        sys.exit()


    #M
    #M set fontsizes according to OS type - Windows fonts are bigger than Mac or
    #M linux
    #M
    #OS = computer
    #if strcmp(OS, 'PCWIN')     #M Windows
    #  fontsize1 = 13
    #  fontsize2 = 15
    #elseif strcmp(OS, 'MACI')  #M Mac (Intel)
    #  fontsize1 = 16
    #  fontsize2 = 18
    #  scrsz = get(0, 'ScreenSize')
    #  if scrsz(3) < 1400
    #    fontsize1 = 12
    #    fontsize2 = 14
    ##elseif strcmp(OS, 'GLNXA64') | strcmp(OS, 'GLNX86')   #M Linux
    # fontsize1 = 14
    #  fontsize2 = 16
    #else                       #M default
    #  fontsize1 = 16
    #  fontsize2 = 18
    #end
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi= screen.physicalDotsPerInch()
    scrsz = screen.geometry()
    scrsz_height = int(scrsz.height() * 0.6)
    scrsz_width = int(scrsz.width() * 0.9)
    op_sys = platform.system()
    screen_height_in = int(scrsz_height/dpi)  # leave 1/2 inch margins
    screen_width_in = int(scrsz_width/dpi)  # leave 1/2 inch margins
    # vert_label_corr is to center height label
    if op_sys == 'Windows':
        fontsize1 = 13
        fontsize2 = 15
        vert_label_corr = 0
    elif op_sys == 'Darwin':   # True is Apple Mac
        fontsize1 = 16
        fontsize2 = 18
        vert_label_corr = 0
        if scrsz.width() < 1400:
            fontsize1 = 12
            fontsize2 = 14
            vert_label_corr = 0
    elif op_sys == 'Linux':
        fontsize1 = 10
        fontsize2 = 12
        vert_label_corr = 100
    else:                    # default
        fontsize1 = 16
        fontsize2 = 16
        vert_label_corr = 0
    #M
    #M initialize the figure
    #M
    #scrsz = get(0, 'ScreenSize')
    max_range = end_range - start_range
    
    #M determine how the display window should be sized
    # set(gcf, 'units', 'pixel')
    # ypos = scrsz(4)*0.25
    ypos = scrsz.height() * 0.25
    # xsize = scrsz(3)*0.95
    xsize = scrsz.width() * 0.95
    # ysize = scrsz(4)*0.5
    ysize = scrsz.height() * 0.5
    
    #M if using softeare opengl then limit the size of the window to avoid
    #M large frame buffer issues
    # opengl_info = opengl('data')
    # if opengl_info.Software
    #   if xsize > 1600
    #     shrink_factor = 1600 ./ xsize
    #     xsize = xsize .* shrink_factor
    #     ysize = ysize .* shrink_factor
    #   end
    # end
    # set(gcf, 'Position', [1 ypos xsize ysize])
        
    
    #M
    #M set up the axes and plot the ionosphere
    #M
    
    #M convert the coodinate frame to curved Earth geometry
    max_range_idx = max_range / range_inc + 1
    # print(max_range, start_range, end_range, range_inc)
    # print(start_height, end_height, height_inc)
    rad_earth = 6371.  # earth radius in km
    # r = rad_earth + [start_height : height_inc : end_height]'
    # gnd_range = [0 : range_inc : max_range]
    # theta = (gnd_range - max_range/2) ./ rad_earth
    # iono_X = r * sin(theta)
    # iono_Y = r * cos(theta)
    r = np.add(rad_earth, np.linspace(start_height, end_height, 
                    int(((end_height - start_height) + 1) / height_inc) + 1))
    gnd_range = np.linspace(0, max_range, int((max_range +1) / range_inc) + 1)
    theta = np.divide( np.subtract(gnd_range, max_range /2), rad_earth)
    rt = r.reshape((-1, 1))  # doing the transpose
    iono_X = np.multiply(rt, np.sin(theta))
    iono_Y = np.multiply(rt, np.cos(theta))
    # print(iono_X.shape, iono_Y.shape, iono_grid.shape)
    
    #M plot the ionospheric slice
    # handle = pcolor(iono_X, iono_Y, iono_grid)
    # shading flat
    # axis equal
    fig = plt.figure(figsize=(screen_width_in, screen_height_in))
    # ax = fig.add_subplot(111)
    ax = fig.add_axes([0, 0.25, .95, 0.5])
    l, b, w, h = ax.get_position().bounds
    ax.axis('off')  # turn off rectangler suround box and tic marks
    image = plt.pcolormesh(iono_X, iono_Y, iono_grid,shading='gouraud')
    ax.set_aspect('equal')
    ax.axis('off')  # turn off rectangler suround box and tic marks
    # plt.show(block=False)
    # plt.savefig('plot.png')
    # plt.draw()
    # plt.pause(0.001)
    l, b, w, h = ax.get_position().bounds
    #M set the axis to take up most of the horizontal extent - leave some space
    #M for margin, ticks and lables
    # min_X = min(min(iono_X))
    min_X = iono_X.min()
    # max_X = max(max(iono_X))
    max_X = iono_X.max()
    # min_Y = min(min(iono_Y))
    min_Y = iono_Y.min()
    # max_Y = max(max(iono_Y))
    max_Y = iono_Y.max()
    # hspace_for_ticks = (max_X - min_X) ./ 40
    hspace_for_ticks = (max_X - min_X) / 40
    # vspace_for_ticks = (max_Y - min_Y) ./ 25
    vspace_for_ticks = (max_Y - min_Y) / 25
        # set(gca, 'Xlim', [min_X - hspace_for_ticks, max_X], ...
      # 'Ylim', [min_Y - vspace_for_ticks, max_Y], ...
      # 'units', 'normal', 'position', [0.01 0.01 0.98 0.98], 'visible', 'Off')
    #plt.xlim(min_X - hspace_for_ticks, max_X)
    #plt.ylim(min_Y - vspace_for_ticks, max_Y)
    #M find horizontal size of axis in pixels and calculate data-to-pixel ratio
    # set(gca,'units','pixels')   
    # pos_vec_pixels = get(gca, 'position')
    l, b, w, h = ax.get_position().bounds
    w_pixel = scrsz.width() * w
    # pix_ratio = pos_vec_pixels(3) ./ (max_X - min_X)
    pix_ratio = w / (max_X - min_X)
    #M determine the vertical size of axis in pixels
    # pos_vec_pixels(4) = pix_ratio .* (max_Y - min_Y)
    # print(l, b, w, h, pix_ratio)
    pixels_height = pix_ratio * (max_Y - min_Y)
    # pos_vec_pixels(2) = 100                #M leave space for colourbar
    b_pixel = scrsz.height() * b
    pixels_bottom = 100
    # set(gca,'position', pos_vec_pixels)
    # ax.set_position([l, pixels_bottom / scrsz.height(), w, pixels_height / 
    #                scrsz.height()])
    
    #M determine the vertical size of figure in pixels required to fit axes,
    #M colorbar and a margin and set the figure accordingly
    # top = pos_vec_pixels(2) + pos_vec_pixels(4)
    top = pixels_bottom + pixels_height
    # fig_pos = get(gcf,'position')
    l, b, w, h = ax.get_position().bounds
    pos_height = top + 15
    # set(gcf, 'Position', fig_pos)
    # ax.set_position([l, b, w, pos_height / scrsz.height()])
    #M handle of the axes
    # axis_handle = gca
        #M
    #M display ground-range ticks
    #M
    # acceptable_tick_stepsize = [100 150 200 250 500 1000]
    acceptable_tick_stepsize = np.array([100, 150, 200, 250, 500, 1000])
    # tick_stepsize = max_range ./ 8
    tick_stepsize = max_range / 8
    # [diff, pp]  = min(abs(acceptable_tick_stepsize - tick_stepsize))
    pp = np.argmin(np.abs(acceptable_tick_stepsize - tick_stepsize))
    # tick_stepsize = acceptable_tick_stepsize(pp)
    tick_stepsize = acceptable_tick_stepsize[pp]

    # tick_gndrng = [0 : fix(max_range ./ tick_stepsize)] .* tick_stepsize
    tick_gndrng = np.multiply(np.linspace(0, int(max_range / tick_stepsize),
                                  int(max_range / tick_stepsize) + 1),
                                tick_stepsize)
    # tick_theta =  (tick_gndrng - max_range/2) ./ rad_earth
    tick_theta = np.divide(np.subtract(tick_gndrng, max_range / 2), rad_earth)
    # tick_len = (max_range ./ 30000) .* 200
    tick_len = (max_range / 30000) * 200
    # hold on  # this is the defalut status in Python
    tick_r = rad_earth + start_height
    # for idx = 1:length(tick_theta)
    for idx in range(0, len(tick_theta)):
        # tick_X1 = tick_r .* sin(tick_theta(idx))
        # tick_X2 = (tick_r - tick_len) .* sin(tick_theta(idx))
        # tick_Y1 = tick_r .* cos(tick_theta(idx))
        # tick_Y2 = (tick_r - tick_len) .* cos(tick_theta(idx))
        # plot([tick_X1 tick_X2], [tick_Y1 tick_Y2], 'k', 'LineWidth', 2)
        tick_X1 = tick_r * np.sin(tick_theta[idx])
        tick_X2 = (tick_r - tick_len) * np.sin(tick_theta[idx])
        tick_Y1 = tick_r * np.cos(tick_theta[idx])
        tick_Y2 = (tick_r - tick_len) * np.cos(tick_theta[idx])
        #plot([tick_X1 tick_X2], [tick_Y1 tick_Y2], 'k', 'LineWidth', 2)
        xpts = np.array([tick_X1, tick_X2])
        ypts = np.array([tick_Y1, tick_Y2])
        plt.plot(xpts, ypts, 'k', linewidth=2)
        tick_label_X = (tick_r - 3 * tick_len) * np.sin(tick_theta[idx])
        tick_label_Y = (tick_r - 3 * tick_len) * np.cos(tick_theta[idx])
        tick_label = str(int(tick_gndrng[idx] + start_range))
        plt.text(tick_label_X, tick_label_Y, tick_label, horizontalalignment=
             'center', fontsize=fontsize1)
    #M display the 'ground range - axis' label
    text_theta = 0
    xlabel_X = rad_earth * np.sin(text_theta)
    xlabel_Y = rad_earth * np.cos(text_theta) - tick_len * 6
    
    plt.text(xlabel_X, xlabel_Y, 'Ground Range (km)', fontsize=fontsize2, 
             horizontalalignment='center')
         
    #M
    #M display the height ticks
    #M
    num_ticks = int(75 * (end_height - start_height) / max_range)
    num_ticks = np.min(np.array([9, num_ticks]))
    num_ticks = np.max(np.array([2, num_ticks]))
    
    acceptable_tick_stepsize = np.array([50, 100, 200, 250, 300,
                                         400, 500, 600, 1000])
    tick_stepsize = (end_height - start_height) / (num_ticks - 1)
    # [diff, pp]  = min(abs(acceptable_tick_stepsize - tick_stepsize))
    pp  = np.argmin(np.abs(acceptable_tick_stepsize - tick_stepsize))
    # tick_stepsize = acceptable_tick_stepsize(pp)
    tick_stepsize = acceptable_tick_stepsize[pp]
    # if ((num_ticks - 1) .* tick_stepsize < end_height)
    if ((num_ticks - 1) * tick_stepsize) < end_height:
    # if ((num_ticks - 1) .* tick_stepsize < end_height - tick_stepsize)
        if ((num_ticks - 1) * tick_stepsize) < end_height - tick_stepsize:
       # if pp < length(acceptable_tick_stepsize)
            if pp < len(acceptable_tick_stepsize):
               # tick_stepsize = acceptable_tick_ste)psize(pp+1)
               tick_stepsize = acceptable_tick_stepsize[pp+1]
        else:
             num_ticks = num_ticks + 1
    
    # while ((num_ticks - 1) .* tick_stepsize > end_height) 
      # num_ticks = num_ticks - 1
    # end
    while ((num_ticks - 1) * tick_stepsize) > end_height:
      num_ticks = num_ticks - 1
     
    # tick_theta =  (0 - max_range/2) ./ rad_earth
    # tick_len = max_range ./ 150
    
    # for idx = 0:num_ticks-1
      # tick_X1 = (rad_earth + idx .* tick_stepsize) .* sin(tick_theta)
      # tick_X2 = tick_X1 - tick_len .* cos(abs(tick_theta))
      # tick_Y1 = (rad_earth + idx .* tick_stepsize) .* cos(tick_theta)
      # tick_Y2 = tick_Y1 - tick_len .* sin(abs(tick_theta))
      # plot([tick_X1 tick_X2], [tick_Y1 tick_Y2], 'k', 'LineWidth', 2)
      
      # tick_label = num2str(tick_stepsize .* idx)
      # tick_label_X = tick_X2 - tick_len ./ 2
      # tick_label_Y = tick_Y2
      # text(tick_label_X, tick_label_Y, tick_label, 'HorizontalAlignment', ...
           # 'right', 'fontsize', fontsize1)
    # end
    tick_theta =  (0 - max_range / 2) / rad_earth
    tick_len = max_range / 150
    
    for idx in range(0, num_ticks):
        tick_X1 = (rad_earth + idx * tick_stepsize) * np.sin(tick_theta)
        tick_X2 = tick_X1 - tick_len * np.cos(np.abs(tick_theta))
        tick_Y1 = (rad_earth + idx * tick_stepsize) * np.cos(tick_theta)
        tick_Y2 = tick_Y1 - tick_len * np.sin(np.abs(tick_theta))
        xpts = np.array([tick_X1, tick_X2])
        ypts = np.array([tick_Y1, tick_Y2])
        plt.plot(xpts, ypts, 'k', linewidth=2)
        tick_label = str(tick_stepsize * idx)
        tick_label_X = tick_X2 - tick_len / 2
        tick_label_Y = tick_Y2
        # print(tick_label,tick_label_X,tick_label_Y,tick_theta)
        plt.text(tick_label_X, tick_label_Y, tick_label, horizontalalignment=
           'right', fontsize=fontsize1)
      #M display the 'height - axis' label
      # text_theta = (- max_range/2) ./ rad_earth
      # text_rot = -text_theta * 180/pi + 90
    
      # pos_adjust = tick_len .* ((end_height - start_height) ./ 400 + 5)
      # ylabel_X = (rad_earth + (end_height - start_height) ./ 2) .* sin(text_theta) ...
        # - pos_adjust .* cos(abs(tick_theta))
      # ylabel_Y = (rad_earth + (end_height - start_height) ./ 2) .* cos(text_theta) ...
        # - pos_adjust .* sin(abs(tick_theta))
    
       # text(ylabel_X, ylabel_Y, 'Altitude (km)', 'rotation', text_rot, ...
         # 'HorizontalAlignment', 'center', 'fontsize', fontsize2
    # the constant 0.007 is to increase speration of label from values
    text_theta = (0 - max_range / 2) / rad_earth
    text_rot = -text_theta * 180 / np.pi + 90
    
    pos_adjust = tick_len * ((end_height - start_height) / 400 + 7.5) # 5
    r_dist = (rad_earth + (end_height - start_height) / 2) - vert_label_corr
    ylabel_X = r_dist * np.sin(text_theta)  - pos_adjust \
                   * np.cos(np.abs(tick_theta))
    ylabel_Y = r_dist * np.cos(text_theta) - pos_adjust \
                   * np.sin(np.abs(tick_theta))
    # print(text_theta,text_rot,pos_adjust,ylabel_X,ylabel_Y)
    
    plt.text(ylabel_X, ylabel_Y, 'Altitude (km)', rotation=text_rot,
         horizontalalignment='center', fontsize=fontsize2)
    
    #M
    #M set up the colourbar
    #M
    #  cb_pos = zeros(1, 4)
    # cb_pos(1) = fig_pos(3) / 5
    # cb_pos(2) = 65
    # cb_pos(3) = fig_pos(3) * 3 / 5
    # cb_pos(4) = 20
    # cb_h = colorbar('SouthOutside', 'Position', cb_pos, 'units', 'pixels')
    # set(cb_h, 'Position', cb_pos, 'units', 'pixel', 'fontsize', fontsize1, ...
      #   'xaxislocation', 'bottom')
    # set(get(cb_h,'xlabel'), 'string', 'Plasma Frequency (MHz) ', 'fontsize', ...
      #   fontsize2)
    save_pos = ax.get_position().bounds
    fig.colorbar(image, ax=ax, orientation='horizontal', shrink=0.45, 
                 aspect=50, label='Plasma Freqency (MHz)')
    new_pos = ax.get_position().bounds
    ax.set_position(save_pos)
    #M
    #M now plot the rays
    #M
    # ray_handle = []
    # for idx = 1:length(ray)
      
      # if ~isempty(ray(idx).gndrng)
        
        # #M resample ray at a finer step size
        # len = length(ray(idx).gndrng)
        # ray_gndrng = [ray(idx).gndrng(1) : 0.1 : ray(idx).gndrng(len)]
        # ray_height = interp1(ray(idx).gndrng, ray(idx).height, ray_gndrng, 'pchip')
    
        #M mask out the ray where it lies outside the ionosphere image
        # mask_idx = find(ray_gndrng < start_range  | ray_gndrng > end_range)
        # ray_gndrng(mask_idx) = NaN
        # mask_idx = find(ray_height < start_height | ray_height > end_height)
        # ray_height(mask_idx) = NaN
    
        #M determine the coodinates of the ray in the image and plot it
        # ray_r = ray_height + rad_earth
        # ray_theta = (ray_gndrng - start_range - max_range/2) ./ rad_earth
        # ray_X = ray_r .* sin(ray_theta)
        # ray_Y = ray_r .* cos(ray_theta)
        
        #M set up the plot command
        # plot_cmd = 'h = plot(ray_X, ray_Y '
        # for ii = 1:length(varargin)
          # if ischar(cell2mat(varargin(ii)))
    	# plot_cmd = [plot_cmd ', ''' cell2mat(varargin(ii)) '''']
          # else
    	# plot_cmd = [plot_cmd ', [' num2str(cell2mat(varargin(ii))) ']']
          # end
        # end
        # plot_cmd = [plot_cmd ')']
        
        #M execute the plot command - catch and throw any exceptions
        # try
          # eval(plot_cmd)
          # ray_handle = [ray_handle, h]
        # catch ME
          # error(ME.message)
        # end
        
      # end
      
    # end
    
    # hold off
    #M
    #M now plot the rays
    #M
    ray_handle = []
    # not the code below assumes that the loop will not execute  print(ray[idx].keys())is ray is
    # an empy list
    for idx in range(len(ray)):
    #for idx in range(len(ray)-1,len(ray)): # for testing purpose only do one ray at first 
      # import ipdb; ipdb.set_trace()
      if len(ray[idx]['gndrng'])!=0:
        # print(len(ray[idx]['gndrng']))
        #M resample ray at a finer step size
        no_points = int((ray[idx]['gndrng'][-1] 
                     - ray[idx]['gndrng'][0]) /
                        0.1) + 1
        ray_gndrng = np.linspace(ray[idx]['gndrng'][0], 
                                 ray[idx]['gndrng'][-1], no_points)
        ray_height = np.interp(ray_gndrng, ray[idx]['gndrng'],
                               ray[idx]['height'])
    
        #M mask out the ray where it lies outside the ionosphere image
        ray_gndrng[ray_gndrng < start_range] = np.nan
        ray_gndrng[ray_gndrng > end_range] = np.nan
        ray_height[ray_height < start_height] = np.nan
        ray_height[ray_height > end_height] = np.nan
        #M determine the coodinates of the ray in the image and plot it
        ray_r = ray_height + rad_earth
        ray_theta = (ray_gndrng - start_range - max_range/2) / rad_earth
        ray_X = ray_r * np.sin(ray_theta)
        ray_Y = ray_r * np.cos(ray_theta)
        plt.plot(ray_X, ray_Y)
        #M set up the plot command
        # plot_cmd = 'h = plot(ray_X, ray_Y '
        # for ii = 1:length(varargin)
          # if ischar(cell2mat(varargin(ii)))
    	# plot_cmd = [plot_cmd ', ''' cell2mat(varargin(ii)) '''']
          # else
    	# plot_cmd = [plot_cmd ', [' num2str(cell2mat(varargin(ii))) ']']
          # end
        # end{'linewidth': 2, 'color': 'k'}
        # plot_cmd = [plot_cmd ')']
        
        #M execute the plot command - catch and throw any exceptions
        # try
          # eval(plot_cmd)
          # ray_handle = [ray_handle, h]
        # catch MEprint(ray_handle)
          # error(ME.message)
        # end
        
      # end
      
    # end
    
    # hold off
    
    # plt.draw()
    # plt.pause(0.001)
    # plt.show()

        plot_cmd = 'plt.plot(ray_X, ray_Y'
        for idx in kwargs.keys():
            if type(kwargs[idx]) == str:
                plot_cmd = plot_cmd + ',' + idx + '=' +\
                    "'" + kwargs[idx] + "'"
            else:
                plot_cmd = plot_cmd + ',' + idx + '=' + str(kwargs[idx])
        plot_cmd = plot_cmd + ')'
        h = eval(plot_cmd)
        ray_handle.append(h)
        
        import os

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results/')
        sample_file_name = "sample10"

        if not os.path.isdir(results_dir):
          os.makedirs(results_dir)

        # plt.savefig(results_dir + sample_file_name)
      # plt.show()
    return ax, ray_handle
    
