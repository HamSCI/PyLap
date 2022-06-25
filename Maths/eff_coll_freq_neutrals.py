#!/usr/bin/env python3
#M
#M Name :
#M   eff_coll_freq_neutrals.m
#M 
#M Purpose / description:
#M   Calculates the effective collision frequency between electrons and
#M   particular neutral atmospheric species. Uses data from "Effective
#M   collision frequency of electrons in gases", Y. Itikawa, Physics of
#M   Fluids, vol 16 no. 6, pp 831-835, 1973.
#M
#M Calling sequence:
#M   coll_freq = eff_coll_freq(T_e, number_density, species)
#M
#M Inputs:
#M   T_e             - 1xN array of electron temperatures, in K
#M   number_density  - 1xN array of number density of neutral species, in cm^-3
#  The index below are Matlab index values. for Python subtract 1
#M   species         - a number specifying the neutral species:
#M                      1  - N_2, nitrogen
#M                      2  - O_2, oxygen
#M                      3  - NO, nitric oxide
#M                      4  - H_20, water vapour
#M                      5  - CO_2, carbon dioxide
#M                      6  - CH_4, methane
#M                      7  - H, hydrogen
#M                      8  - He, helium
#M                      9  - O, oxygen
#M                      10 - Ar, argon
#M
#M Output:
#M   coll_freq  - the effective electron-neutral collision frequency (Hz)
#M                for the input neutral species
#M
#M   V1.0  L.H. Pederick  31/10/2012
#M     Initial version.
#M
#M   V1.1  M.A. Cervera  18/05/2018
#M     Minor code and comment tidy-ups
#M
#
#    W. C. Liles 29/06/2020
#       Convert to Python
#
#
import sys
import numpy as np
from scipy.interpolate import interp1d
#

#function coll_freq = eff_coll_freq_neutrals(T_e, number_density, species)
#    persistent T_e_axis coll_freq_data
def eff_coll_freq_neutrals(T_e, number_density, species):  
    #T_e = T_e[:len(number_density)]
    #import ipdb; ipdb.set_trace()
    if len(T_e) != len(number_density):
      print('Inputs 1 and 2 must have the same size')
      sys.exit('eff_coll_freq_neutrals')
    #import ipdb; ipdb.set_trace()

    if ~np.all(np.isreal(T_e)) and ~np.all(np.isreal(number_density)):
      print('Inputs 1 and 2 must be numeric and real')
      sys.exit('eff_coll_freq_neutrals') 
    try:
        eff_coll_freq_neutrals.used += 1
    except AttributeError:
        eff_coll_freq_neutrals.used = 1
        #M Electron temperature array for the Itikawa (1973) electron-neutral
        #M collision frequency data defined in the coll_freq_data array below.
        eff_coll_freq_neutrals.T_e_axis = np.array(
            [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500,
             3000, 3500, 4000, 4500, 5000])
        
        
        #M Electron-neutral collision frequency data from Itikawa (1973). Data is 
        #M the effective collision frequency as a function of species and electron
        #M temerature for unitiary number density of the neutral species.
        eff_coll_freq_neutrals.coll_freq_data = np.array([
               np.array([0.255, 0.491, 0.723, 0.948, 1.17, 2.11,
                         2.86, 3.49, 4.08, 4.72, 5.44, 6.23,
                         7.08, 7.96]),   #M N_2
               np.array([0.123, 0.237, 0.335, 0.425, 0.512, 0.956, 1.41,
                         1.87, 2.32, 2.72, 3.08, 3.40, 3.68, 3.92]),   #M O_2
               np.array([0.589, 0.528, 0.566, 0.673, 0.859, 2.73, 4.88,
                         6.46, 7.49, 8.13, 8.54, 8.79, 8.95, 9.06]),   #M NO
               np.array([158., 107., 84.9, 71.5, 62.3, 39.3, 29.3, 23.7,
                         20.1, 17.6, 15.8, 14.5, 13.4, 12.5]),   #M H_2O
               np.array([10.1, 10.1, 10.1, 9.96, 9.76, 8.04, 6.45, 5.34,
                         4.61, 4.14, 3.86, 3.71, 3.68, 3.74]),   #M CO_2
               np.array([1.51, 0.987, 0.719, 0.568, 0.480, 0.396, 0.486, 0.636,
                          0.822, 1.04, 1.29, 1.58, 1.91, 2.26]),   #M CH_4         
               np.array([3.44, 4.85, 5.89, 6.72, 7.42, 9.81, 11.3, 12.2, 12.9,
                         13.4, 13.8, 14.1, 14.3, 14.5]),   #M H
               np.array([0.448, 0.654, 0.820, 0.963, 1.09, 1.62, 2.05, 2.41,
                         2.74, 3.03, 3.30, 3.55, 3.78, 4.00]),   #M He
               np.array([0.081, 0.138, 0.192, 0.245, 0.297, 0.551, 0.800,
                         1.04, 1.28, 1.51, 1.73, 1.94, 2.14, 2.34]),   #M O
               np.array([0.341, 0.293, 0.243, 0.203, 0.173, 0.110, 0.126,
                       0.185, 0.272, 0.379, 0.504, 0.643, 0.795, 0.960]) #M Ar
                ]) * 1e-8
        
    
    #M Calculate the electron-neutral collision frequency for the input
    #M neutral species
#M    coll_freq = lininterp1f(T_e_axis, coll_freq_data(species, :), T_e, ...
#M	                    NaN) .* number_density
    #import ipdb; ipdb.set_trace()
    from scipy import interpolate
    coll_freq = interpolate.interp1d(eff_coll_freq_neutrals.T_e_axis, 
                          eff_coll_freq_neutrals.coll_freq_data[species])
    freq=[]
    freq.append(coll_freq(T_e))
    freq*number_density
    #import ipdb; ipdb.set_trace()

    return freq