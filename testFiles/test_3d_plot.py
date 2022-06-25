#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:35:55 2020

@author: william
"""

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio

mpl.rcParams['legend.fontsize'] = 10
data=sio.loadmat('/home/alex/MATLAB/projects/eclipsesim/sami3/grid.mat')
#print(data.get('grid_lats')[0][0][0])
lat= data.get('grid_lats')[0][0]
lon= data.get('grid_lons')[0][0]
height= data.get('grid_heights')[0][0]
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(lat, np.mod(lon, 360), height, '+r',label='parametric curve')
ax.legend()
plt.show()