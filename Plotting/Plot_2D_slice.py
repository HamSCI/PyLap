#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:54:40 2020

@author: william

this is a class to aid in PHaRLAP 2D displays
it is bassed on plot_ray_iono_slice.m

It enables varialbes to be saved between using methods
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import platform
from qtpy.QtWidgets import QApplication

class Plot_2D_slice:
    __rad_earth  = 6371.  # earth radius in km
    
    # set up operating system and screen parameters
    # create an object of this class
    # determine all machine and operating system dependicies
    # such as font sizes and size of image on scree
    def __init__(self):
        matplotlib.use('TkAgg')
        app = QApplication(sys.argv)
        screen = app.screens()[0]
        dpi= screen.physicalDotsPerInch()
        scrsz = screen.geometry()
        self.__scrsz_height = int(scrsz.height() * 0.6)
        self.__scrsz_width = int(scrsz.width() * 0.9)
        op_sys = platform.system()
        self.__screen_height_in = int(self.__scrsz_height/dpi)  # leave 1/2 inch margins
        self.__screen_width_in = int(self.__scrsz_width/dpi)  # leave 1/2 inch margins
        # vert_label_corr is to center height label
        if op_sys == 'Windows':
            self.__fontsize1 = 13
            self.__fontsize2 = 15
            self.__vert_label_corr = 0
        elif op_sys == 'Darwin':   # True is Apple Mac
            self.__fontsize1 = 16
            self.__fontsize2 = 18
            self.__vert_label_corr = 0
            if scrsz.width() < 1400:
                self.__fontsize1 = 12
                self.__fontsize2 = 14
                self.__vert_label_corr = 0
        elif op_sys == 'Linux':
            self.__fontsize1 = 10
            self.__fontsize2 = 12
            self.__vert_label_corr = 100
        else:                    # default
            self.__fontsize1 = 16
            self.__fontsize2 = 16
            self.__vert_label_corr = 0
        #M determine how the display window should be sized
        #self.__ypos = self.__scrsz.height() * 0.25
            self.__xsize = self.__scrsz.width() * 0.95
            self.__ysize = self.__scrsz.height() * 0.5
        
    #
    # set up the axes to the ionosphere and rays
    #
    def set_boundries(self, start_range, end_range, range_inc, start_height,
                      end_height, height_inc):
        max_range = end_range - start_range
        heights = np.linspace(start_height, end_height, int(((
            end_height - start_height) + 1) / height_inc))
        no_heights = heights.size + 1
        ranges = np.linspace(start_range, end_range, int(((end_range - start_range)
                         + 1) / range_inc))
        no_ranges = ranges.size + 1
        #M convert the coodinate frame to curved Earth geometry
        max_range_idx = max_range / range_inc + 1
        # print(max_range, start_range, end_range, range_inc)
        # print(start_height, end_height, height_inc)

        r = np.add(Plot_2D_slice.__rad_earth, np.linspace(start_height, 
                        end_height, int(((end_height - start_height) + 1)
                        / height_inc) + 1))
        gnd_range = np.linspace(0, max_range, int((max_range +1) / range_inc) + 1)
        theta = np.divide( np.subtract(gnd_range, max_range /2), 
                          Plot_2D_slice.__rad_earth)
        rt = r.reshape((-1, 1))  # doing the transpose
        self.__iono_X = np.multiply(rt, np.sin(theta))
        self.__iono_Y = np.multiply(rt, np.cos(theta))
        self.__end_height = end_height
        self.__start_height = start_height
        self.__start_range = start_range
        self.__end_range = end_range
        self.__max_range = max_range
        self.__ranges = ranges
        self.__no_ranges = no_ranges
        self.__no_heights = no_heights
        
    # display the ionosphere grid along with tic marks
    # show with earth curvature   
    def set_iono_grid(self,iono_grid):
        
        #M plot the ionospheric slice
        iono_grid_size = iono_grid.shape
        if self.__no_heights != iono_grid_size[0]:
            print('start_height, end_height and height_inc inconsistent '
                  'with iono_grid in plot_ray_iono_slice')
            sys.exit()

        if self.__no_ranges != iono_grid_size[1]:
            print('start_range, end_range and range_inc inconsistent with '
                  'iono_grid in plot_ray_iono_slice')
            sys.exit()
        # handle = pcolor(iono_X, iono_Y, iono_grid)
        # shading flat
        # axis equal
        fig = plt.figure(figsize=(self.__screen_width_in,
                                  self.__screen_height_in))
        ax = fig.add_axes([0, 0.25, .95, 0.5])
        l, b, w, h = ax.get_position().bounds
        ax.axis('off')  # turn off rectangler suround box and tic marks
        image = plt.pcolormesh(self.__iono_X, self.__iono_Y, iono_grid,
                               shading='gouraud')
        ax.set_aspect('equal')
        ax.axis('off')  # turn off rectangler suround box and tic marks
        # plt.show(block=False)
        # plt.savefig('plot.png')
        # plt.draw()
        # plt.pause(0.001)
        l, b, w, h = ax.get_position().bounds
        #M set the axis to take up most of the horizontal extent - leave some space
        #M for margin, ticks and lables
        min_X = self.__iono_X.min()
        max_X = self.__iono_X.max()
        min_Y = self.__iono_Y.min()
        max_Y = self.__iono_Y.max()
        # hspace_for_ticks = (max_X - min_X) / 40
        # vspace_for_ticks = (max_Y - min_Y) / 25

        l, b, w, h = ax.get_position().bounds
        # w_pixel = self.__scrsz.width() * w
        pix_ratio = w / (max_X - min_X)
        #M determine the vertical size of axis in pixels

        pixels_height = pix_ratio * (max_Y - min_Y)
        #M leave space for colourbar
        # b_pixel = self.__scrsz.height() * b
        pixels_bottom = 100

        
        #M determine the vertical size of figure in pixels required to fit axes,
        #M colorbar and a margin and set the figure accordingly
        # top = pos_vec_pixels(2) + pos_vec_pixels(4)
        # top = pixels_bottom + pixels_height

        l, b, w, h = ax.get_position().bounds
        # pos_height = top + 15

        #M handle of the axes

        #M display ground-range ticks
        #M
        acceptable_tick_stepsize = np.array([100, 150, 200, 250, 500, 1000])
        tick_stepsize = self.__max_range / 8
        pp = np.argmin(np.abs(acceptable_tick_stepsize - tick_stepsize))
        tick_stepsize = acceptable_tick_stepsize[pp]
    
        tick_gndrng = np.multiply(np.linspace(0, int(self.__max_range / 
                                    tick_stepsize),
                                    int(self.__max_range / tick_stepsize) + 1),
                                    tick_stepsize)
        tick_theta = np.divide(np.subtract(tick_gndrng, self.__max_range / 2),
                               Plot_2D_slice.__rad_earth)
        tick_len = (self.__max_range / 30000) * 200
        tick_r = Plot_2D_slice.__rad_earth + self.__start_height

        for idx in range(0, len(tick_theta)):

            tick_X1 = tick_r * np.sin(tick_theta[idx])
            tick_X2 = (tick_r - tick_len) * np.sin(tick_theta[idx])
            tick_Y1 = tick_r * np.cos(tick_theta[idx])
            tick_Y2 = (tick_r - tick_len) * np.cos(tick_theta[idx])
            
            xpts = np.array([tick_X1, tick_X2])
            ypts = np.array([tick_Y1, tick_Y2])
            plt.plot(xpts, ypts, 'k', linewidth=2)
            tick_label_X = (tick_r - 3 * tick_len) * np.sin(tick_theta[idx])
            tick_label_Y = (tick_r - 3 * tick_len) * np.cos(tick_theta[idx])
            tick_label = str(int(tick_gndrng[idx] + self.__start_range))
            plt.text(tick_label_X, tick_label_Y, tick_label,
                     horizontalalignment='center', fontsize=self.__fontsize1)
        #M display the 'ground range - axis' label
        text_theta = 0
        xlabel_X = Plot_2D_slice.__rad_earth * np.sin(text_theta)
        xlabel_Y = Plot_2D_slice.__rad_earth * np.cos(text_theta) - tick_len * 6
        
        plt.text(xlabel_X, xlabel_Y, 'Ground Range (km)',
                 fontsize=self.__fontsize2, horizontalalignment='center')
             
        #M
        #M display the height ticks
        #M
        num_ticks = int(75 * (self.__end_height - self.__start_height) \
                        / self.__max_range)
        num_ticks = np.min(np.array([9, num_ticks]))
        num_ticks = np.max(np.array([2, num_ticks]))
        
        acceptable_tick_stepsize = np.array([50, 100, 200, 250, 300,
                                             400, 500, 600, 1000])
        tick_stepsize = (self.__end_height - self.__start_height) \
            / (num_ticks - 1)

        pp  = np.argmin(np.abs(acceptable_tick_stepsize - tick_stepsize))
        tick_stepsize = acceptable_tick_stepsize[pp]
        if ((num_ticks - 1) * tick_stepsize) < self.__end_height:
            if ((num_ticks - 1) * tick_stepsize) < self.__end_height \
                - tick_stepsize:
                if pp < len(acceptable_tick_stepsize):
                   # tick_stepsize = acceptable_tick_ste)psize(pp+1)
                   tick_stepsize = acceptable_tick_stepsize[pp+1]
            else:
                 num_ticks = num_ticks + 1
        

        while ((num_ticks - 1) * tick_stepsize) > self.__end_height:
          num_ticks = num_ticks - 1
         

        tick_theta =  (0 - self.__max_range / 2) / Plot_2D_slice.__rad_earth
        tick_len = self.__max_range / 150
        
        for idx in range(0, num_ticks):
            tick_X1 = (Plot_2D_slice.__rad_earth + idx * tick_stepsize)  \
                * np.sin(tick_theta)
            tick_X2 = tick_X1 - tick_len * np.cos(np.abs(tick_theta))
            tick_Y1 = (Plot_2D_slice.__rad_earth + idx * tick_stepsize) \
                * np.cos(tick_theta)
            tick_Y2 = tick_Y1 - tick_len * np.sin(np.abs(tick_theta))
            xpts = np.array([tick_X1, tick_X2])
            ypts = np.array([tick_Y1, tick_Y2])
            plt.plot(xpts, ypts, 'k', linewidth=2)
            tick_label = str(tick_stepsize * idx)
            tick_label_X = tick_X2 - tick_len / 2
            tick_label_Y = tick_Y2
            # print(tick_label,tick_label_X,tick_label_Y,tick_theta)
            plt.text(tick_label_X, tick_label_Y, tick_label,
                     horizontalalignment='right', fontsize=self.__fontsize1)
          #M display the 'height - axis' label
          
             # 'HorizontalAlignment', 'center', 'fontsize', fontsize2
        # the constant 0.007 is to increase speration of label from values
        text_theta = (0 - self.__max_range / 2) / Plot_2D_slice.__rad_earth
        text_rot = -text_theta * 180 / np.pi + 90
        
        pos_adjust = tick_len * ((self.__end_height - self.__start_height) \
                                 / 400 + 7.5) # 5
        r_dist = (Plot_2D_slice.__rad_earth + (self.__end_height
                  - self.__start_height) \
                  / 2) - self.__vert_label_corr
        ylabel_X = r_dist * np.sin(text_theta)  - pos_adjust \
                       * np.cos(np.abs(tick_theta))
        ylabel_Y = r_dist * np.cos(text_theta) - pos_adjust \
                       * np.sin(np.abs(tick_theta))
        # print(text_theta,text_rot,pos_adjust,ylabel_X,ylabel_Y)
        
        plt.text(ylabel_X, ylabel_Y, 'Altitude (km)', rotation=text_rot,
             horizontalalignment='center', fontsize=self.__fontsize2)
        return ax, fig, image
    
    # 
    #
    # set up the colourbar
    #
    def show_color_bar(self,ax, fig,image):
        save_pos = ax.get_position().bounds
        fig.colorbar(image, ax=ax, orientation='horizontal', shrink=0.45, 
                 aspect=50, label='Plasma Freqency (MHz)')
        new_pos = ax.get_position().bounds
        ax.set_position(save_pos)
    
    #
    # method to display rays
    # note that this method can be called multiple times with the 
    # same iono_grid
    #
    #
    # input
    #    ray   array of rays to be ploted
    #    varargin  these are keyword - value pairs that are passed
    #      to the plot command for each ray plotted
    #
    #  output
    #    ray_handle an array of the handles for each ray plotted
    #      useful if want to change any display parameters of the ray
    
    def show_rays(self, ray, **kwargs):
        for rayID in range(0, len(ray)):
            if 'ground_range' in ray[rayID].keys():
                ray[rayID]['gndrng'] = ray[rayID]['ground_range']
        for ii in range(0, len(ray)):
            # print(ii, ray[ii]['height'], ray[ii]['gndrng'])
            if len(ray[ii]['height']) != \
                    len(ray[ii]['gndrng']):
                print('ray height and ground range vectors have diffent lengths' +
                      ' in plot_ray_iono_slice')
                sys.exit()
        ray_handle = []
        # not the code below assumes that the loop will not execute  print(ray[idx].keys())is ray is
        # an empy list
        for idx in range(len(ray)):
        #for idx in range(len(ray)-1,len(ray)): # for testing purpose only do one ray at first 
          if ray[idx]['gndrng'].any():
            
            #M resample ray at a finer step size
            no_points = int((ray[idx]['gndrng'][-1] 
                         - ray[idx]['gndrng'][0]) /
                            0.1) + 1
            ray_gndrng = np.linspace(ray[idx]['gndrng'][0], 
                                     ray[idx]['gndrng'][-1], no_points)
            ray_height = np.interp(ray_gndrng, ray[idx]['gndrng'],
                                   ray[idx]['height'])
        
            #M mask out the ray where it lies outside the ionosphere image
            ray_gndrng[ray_gndrng < self.__start_range] = np.nan
            ray_gndrng[ray_gndrng > self.__end_range] = np.nan
            ray_height[ray_height < self.__start_height] = np.nan
            ray_height[ray_height > self.__end_height] = np.nan
            #M determine the coodinates of the ray in the image and plot it
            ray_r = ray_height + Plot_2D_slice.__rad_earth
            ray_theta = (ray_gndrng - self.__start_range - self.__max_range/2)\
                / Plot_2D_slice.__rad_earth
            ray_X = ray_r * np.sin(ray_theta)
            ray_Y = ray_r * np.cos(ray_theta)
            plt.plot(ray_X, ray_Y)
            
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
        # plt.show()
        return ray_handle
    
    # map a point in range and height to image coordinates
    # inputs
    #   pt_range   the range of the point in km
    #   pt_height  the height of the point in km
    #   optional arguments are units for angle and
    #   reference plane for angle
    #   the legal values are 'degrees' or 'radians'
    #   and 'horizontal' or 'vertical'
    #   defaults are 'radians' and 'vertical'
    #
    # outputs
    #   x coordinate corresponding to the input point in image coord.
    #   y coordinate corresponding to the input point in image coord.
    #   angle the angle in units specified to reference plane
    def map_point(self, pt_range, pt_height):
        r = pt_height + Plot_2D_slice.__rad_earth
        theta = (pt_range - self.__start_range - self.__max_range/2) \
            / Plot_2D_slice.__rad_earth
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return x, y, theta
    
    #
    # draw a zenth line
    # given a range in km this method draws a line segment
    # perpendicular to the point on the earth surface at the specified
    # range
    #
    # input
    #    ax   pointer to figure
    #    earth_range   rane to point on earth surface
    #    dist_low  start distance of line relative to earth in km
    #    dist_high  end distance of line relative to earth in km
    #    varargin
    #     angle measuremnt units 'radians' or 'degrees'
    #     reference plane for angle 'vertical' or 'horizontal'
    #     the rest of the arguments are keyword/value pairs passed to
    #     the plot function
    # note: if angle measurment or reference plane are specified, they 
    # must be specified before the keyword/vlaue pairs for plot cmd
    #
    # output
    #   line_handle the handle for the line drawn
    #   x two elelent array of the x value of the start and end point 
    #      of the line in image coordinates
    #   y two elelent array of the y value of the start and end point 
    #      of the line in image coordinates
    #   angle the angle of the line segment in the units and reference
    #     plane specified.  default is radians and vertical
    def zenith_line(self, earth_range, dist_low, dist_hi, **kwargs):
        # dist_low is the bottom of the line to be drawn
        # to place on the earth surface, set to 0
        # to place benath the earth surface set to a negative number in km
        # beneath_ surface
        ray_l_x, ray_l_y, theta_l = self.map_point(earth_range, dist_low)
        ray_h_x, ray_h_y, theta_h = self.map_point(earth_range, dist_hi)
        x = np.array([ray_l_x, ray_h_x])
        y = np.array([ray_l_y, ray_h_y])
        plot_cmd = 'plt.plot(x, y'
        for idx in kwargs.keys():
            if type(kwargs[idx]) == str:
                plot_cmd = plot_cmd + ',' + idx + '=' +\
                        "'" + kwargs[idx] + "'"
            else:
                plot_cmd = plot_cmd + ',' + idx + '=' + str(kwargs[idx])
        plot_cmd = plot_cmd + ')'
        h = eval(plot_cmd)
        return h
    
    # get image coordinates for perpendicular line from speicifed range
    # to max hight
    # given range on earth, returns image coordinate for 
    #  x_earth, y_earth
    #  x_max_hight, y_max_height
    # theta which is angle of line segment from vertical, origan earth center
    def get_im_coords(self, earth_range):
        ray_l_x, ray_l_y, theta_0 = self.map_point(earth_range, 0)
        ray_h_x, ray_h_y, theta_ht = self.map_point(earth_range, 
                                                    self.__end_height)
        print(theta_0)
        print(theta_ht)
        return ray_l_x, ray_l_y, ray_h_x, ray_h_y, theta_0
    
    # get image size in pixels. Useful if want to plot directly on
    # image
    def get_image_size(self):
        xsize = self.__xsize
        ysize = self.__ysize
        return xsize, ysize
         

        
    

        