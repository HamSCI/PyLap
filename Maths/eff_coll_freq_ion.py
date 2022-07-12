#!/usr/bin/env python3
#M
#M Name 
#M   eff_coll_freq_ion.m
#M 
#M Purpose / description
#M   Calculates the effective collision frequency between electrons and ions in 
#M   the Earth's Ionosphere. Based on Schunk, R. W., and A. F. Nagy (1978), 
#M   Electron temperatures in the F region of the ionosphere Theory and 
#M   observations, Rev. Geophys. Space. Phys., 16(3), 355–399.
#M
#M Calling sequence
#M   coll_freq_ion = eff_coll_freq_ion(T_e, T_ion, elec_dens)
#M
#M Inputs
#M   T_e        - array of electron temperatures (K)
#M   T_ion      - array of ion temperatures (K)
#M   elec_dens  - array of number density of electrons (m^-3)
#M
#M Output
#M   coll_freq  - The effective electron-ion collision frequency (Hz)
#M
#M   V1.0  M.A. Cervera  18/05/2018
#M     Initial version.
#M
#
#    01/07/2020  W.C. Liles
#       Convert to Python
#
#
import sys
import numpy as np
#
# function coll_freq_ion = eff_coll_freq_ion(T_e, T_ion, elec_dens)
def coll_freq_ion(T_e, T_ion, elec_dens):
  # print(len(T_e))
  elec_dens=elec_dens[:len(T_e)]
  #print(len(elec_dens))
  #print(len(T_ion))
  if len(T_e) != len(elec_dens):
    print('All inputs must have the same size')
    sys.exit('coll_freq_ion')
  
    
  if len(T_e) != len(T_ion):
    print('All inputs must have the same size')
    sys.exit('coll_freq_ion')
  
  # for  i in range (len(T_e)):
  if (not np.isreal(T_e.any)) or (not np.isreal(T_ion.any)) or \
     (not np.isreal(elec_dens.any)):
     print('All inputs must be numeric and real')
     sys.exit('coll_freq_ion')

  # print (elec_dens)
  ki_sq = 2.09985255e-4 * elec_dens / T_ion
  ke_sq = 2.09985255e-4 * elec_dens / T_e
  # print(ke_sq)
  # print(ki_sq)

  ln_coulomb_int = 13.484870477617616 + np.log(T_e) - 0.5*np.log(ke_sq) - ((ke_sq + ki_sq) / ki_sq) * (0.5 * np.log((ki_sq + ke_sq) / ke_sq))
	       
  coll_freq_ion = 3.63315e-6 * elec_dens * (T_e ** (-3./2.)) * ln_coulomb_int
  return coll_freq_ion


