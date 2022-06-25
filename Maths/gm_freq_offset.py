#!/usr/bin/env python3
#M
#M Name : 
#M   gm_freq_offset.m
#M
#M Purpose :
#M   Calculates the approximate geomagnetic O-X mode frequency split (MHz) for
#M   a specified propagation path. This can then be applied to a ray
#M   calculated using ART, 2D NRT or "no geomagnetic field" 3D NRT to give an
#M   approximate O and X mode solution.
#M
#M Calling sequence :
#M   [del_freq_O, del_freq_X] = ...
#M      gm_freq_offset(lat_cp, lon_cp, ht_cp, ray_bearing, freq, plas_freq, UT)
#M
#M Inputs :
#M   lat_cp      - latitude of control point of ray (i.e. the ray apogee) (deg)  
#M   lon_cp      - longitude of control point of ray (deg)
#M   ht_cp       - height of control point of ray (deg)
#M   ray_bearing - bearing of ray from tx. (deg)
#M   freq        - radio frequency of the ray (MHz)
#M   UT          - universal time, 5x1 array (year, month, day, hour, minute)
#M   plas_freq   - plasma freq of the ionosphere at the ray control point (MHz)
#M
#M Outputs :
#M   del_freq_O  -  shift to be applied to ray frequency for O mode case (MHz)
#M   del_freq_X  -  shift to be applied to ray frequency for X mode case (MHz)
#
#  Routines called
#     igrf2016
#M
#M Notes :
#M   1. The "no geomagnetic field" ray must also have the group path offset
#M      calculated. If group_path, phase_path and freq are the "no field"
#M      values, the the group_path offsets are:
#M
#M      del_group_O = (group_path - phase_path) .* del_freq_O / freq
#M      del_group_X = (group_path - phase_path) .* del_freq_X / freq
#M
#M   2. Application of the calculated group path and frequency shifts is as
#M      follows :
#M
#M      group_O = group + del_group_O
#M      group_X = group + del_group_X
#M
#M      freq_O = freq + del_freq_O
#M      freq_X = freq + del_freq_X
#M
#M   3. see the example code ois_synth.m for an example on how gm_freq_offset.m
#M      is used
#M
#M
#M References :
#M   1. Bennett et al. (1994), "Analytic calculation of the ordinary (O) and 
#M      extraordinary (X) mode nose frequencies on oblique ionograms", J. Atmos.
#M      Terr. Phys., Vol 56, No. 5, pp 631-636.
#M
#M   2. Dyson and Bennett (1980), "A Universal Chart for Use in Relating
#M      Ionospheric Absorption to Phase Path and Group Path", IEEE
#M      Transactions on Antennas and Propagation, Vol AP-28, No. 3, 380-384.
#M
#M   3. Bennett et al. (1991), "Analytic Ray Tracing for the Study of HF
#M      Magneto-ionic Radio Propagation in the ionosphere", Applied
#M      Computational Electromagnetics Society Journal, Vol 6(1), 192-210
#M      (see Appendix)
#M
#M   4. Chen et al. (1990), "Automatic fitting of quasi-parabolic segments to
#M      ionospheric profiles", J. Atmos. Terr. Phys, Vol 52, No. 4, 277-288 
#M      (see Appendix B)
#M
#M
#M Modification history:
#M   V1.0  M.A. Cervera  08/05/2015
#M       Based on Matlab code and algorithm provided by Andrew Heitman
#M
#        W. C. Liles 08/08/2020
#            Convert to Python
#
#
import numpy as np
import sys
import math

from numpy.core.numerictypes import find_common_type
#
from pylap.igrf2016 import igrf2016
#
# function [del_freq_O, del_freq_X] = ...
#      gm_freq_offset(lat_cp, lon_cp, ht_cp, ray_bearing, freq, plas_freq, UT)
def gm_freq_offset(lat_cp, lon_cp, ht_cp, ray_bearing, freq, plas_freq, UT):
			      

    #M check inputs
    # first 6 inputs must be scaler note this ndim test is not the best since
    #   string values also have ndim = 0
    if np.dim(lat_cp) != 0 or np.dim(lon_cp) != 0 or np.dim(ht_cp) != 0 or \
       np.dim(ray_bearing) != 0 or np.dim(freq) != 0 or \
       np.dim(plas_freq) != 0:
           print('Inputs 1-6 must be scalers in gm_freq_offset')
           sys.exit(1)
    
    #M Return frequency offsets of zero when input frequency is zero.
    if freq == 0:
      del_freq_O = 0
      del_freq_X = 0
      return del_freq_O, del_freq_X

    
    #M gyro_factor - converts magnetic field strength (Tesla) to electron
    #M gyro-frequency 
    gyro_factor = 2.7992492071667
    
    #M  geomagnetic field parameters at the control point.
    mag_field = igrf2016.igrf2016(lat_cp, lon_cp, UT, ht_cp)
    B_mag = mag_field[3]
    B_dip = mag_field[7]
    B_dec = mag_field[9]
    
    gyro_freq = B_mag * gyro_factor
    
    #M angle between B-field and ray direction at control point
    theta = np.arccos(np.cos(np.radians(B_dip)) * 
                      np.cos(np.radians(B_dec - ray_bearing)))
    
    
    #M Compute o-mode and x-mode offsets from the effective frequency, as defined
    #M by Bennett, Chen & Dyson (1991). Strictly speaking, 'freq' is assumed to
    #M represent this unperturbed value.
    
    #M X = (fN/f)^2 and Y = fH/f are the normalised frequency variables
    #M W = the ratio of X-1 and Y (negative for HF ionospheric propagation)
    Y = gyro_freq / freq
    X = (plas_freq / freq) ** 2
    W_o = (X - 1) / Y
    W_x = (X - np.sqrt(X)* Y - 1) / Y
    
    #M Calculate o-mode and x-mode h factors at the ray's control point. 
    #M Theta is the angle between the wave normal (i.e. direction of phase
    #M propagation) and the geomagnetic field. As we can't calculate this value 
    #M (full 3D magneto-ionic raytrace is required) we will approximate this value 
    #M with the angle between the ray direction and the geomagnetic field. This
    #M approximation is OK for oblique propagation but errors will grow as the ray
    #M elevation -> 90 deg. 
    h_o = h_param(W_o, theta, +1)
    h_x = h_param(W_x, theta, -1)
    
    #M Calculate the effective frequency offsets.
    g_o = Y * h_o/2
    g_x = Y * h_x/2
    del_freq_O = freq * (g_o / (g_o - 1))  
    del_freq_X = freq * (g_x / (g_x - 1)) 
    
    
    return del_freq_O, del_freq_X

    
    
#M-------------------------------------------------------------------------------

# function h = h_param(W, theta, OX_mode)
def h_param(W, theta, OX_mode):
    #M Equation 7 from Dyson & Bennett (1980).
    
    #M The input argument OX_mode specifies the O (+1) or X mode (-1) version of
    #M the equation. 
    
    #M stay away from theta = 90 deg as denominator -> 0 here for X mode
    idx = find_common_type(theta > np.pi / 2 - 1e-5 and theta < np.pi/2 + 1e-5)
    theta[idx] = theta[idx] - 1e-5
    
    S2 = np.sin(theta) ** 2
    C2 = np.cos(theta) ** 2
    
    fact = math.sqrt(1 + 4 * C2 * W ** 2 / S2 ** 2)
    numerator = 2 * C2 * W * (1 - OX_mode * fact)
    denominator = 1 + C2 + OX_mode * S2 * fact + 4 * C2 * W ** 2 / S2
    h = numerator / denominator
    
    #M at the limits theta -> 0 and theta -> 180 :
    #M    h -> +1 for O mode if W != 0
    #M          0 for O mode if W == 0
    #M    h -> -1 for X mode for all W
    idx = np.where(S2 < 1e-10)
    if OX_mode == -1:
      h[idx] = OX_mode   
    else:
      if W == 0:
        h[idx] = 0
      else:
        h[idx] = 1
    
    return h
    
