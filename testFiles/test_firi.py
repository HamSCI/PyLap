#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:11:38 2020

@author: william
"""
import scipy.io as sio
import numpy as np
def test_firi():
    



    ray_bear = 324.7             #M bearing of rays

    freq = 15.0                  #M ray frequency (MHz)
    data = sio.loadmat('/home/william/Desktop/iri_firi.mat')
    # dict_keys(['__header__', '__version__', '__globals__', 'site_lat',
    # 'site_lon', 'R12', 'UT', 'start_height', 'height_step', 'num_heights',
    # 'iono', 'iono_extra', 'iono_pre', 'height_axis', 'bottom_transition',
    # 'top_transition', 'firi_transition', 'top_idx', 'bottom_idx', 'valid',
    # 'bottom'])
    site_lat = data['site_lat'][0][0]
    start_height = int(data['start_height'][0][0])
    num_heights = int(data['num_heights'][0][0])
    height_step = int(data['height_step'][0][0])
    iono_extra = data['iono_extra'][0]
    
    print(num_heights * height_step)
    height_axis_c = start_height + np.arange(0,num_heights * height_step,
        height_step)
    height_axis = data['height_axis'][0]
    print(height_axis_c)
    print(height_axis)
    print(len(height_axis_c),len(height_axis))
    