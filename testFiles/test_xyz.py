#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:22:26 2020

@author: william
"""

import scipy.io as sio
import numpy as np
import xyz2ENU

def test_xyz():
    data= sio.loadmat('/home/william/Desktop/SagaPos.mat')
    lat = data['SagaPos']['lat'][0][0][0]
    lon = data['SagaPos']['lon'][0][0][0]
    alt = data['SagaPos']['alt'][0][0][0]
    xarr = data['SagaPos']['xarr'][0][0][0]
    yarr = data['SagaPos']['yarr'][0][0][0]
    zarr = data['SagaPos']['zarr'][0][0][0]
    centerlat = data['SagaPos']['centerlat'][0][0][0]
    centerlon = data['SagaPos']['centerlon'][0][0][0]
    xrot = data['SagaPos']['xrot'][0][0][0]
    yrot = data['SagaPos']['yrot'][0][0][0]
    zrot = data['SagaPos']['zrot'][0][0][0]
    e, n, u = xyz2ENU.xyz2ENU(xarr,yarr,zarr,
                              centerlat[0],centerlon[0])
    print(e,n,u,xrot[0],yrot[0],zrot[0])
    print(e,n,u)
    
    