#!/usr/bin/env python3
#M
#M Name :
#M   eff_coll_freq.m
#M 
#M Purpose / description:
#M   Calculates the effective collision frequency between electrons and
#M   atmospheric constituents (both ions and neutrals).
#M
#M Calling sequence:
#M   coll_freq = eff_coll_freq(T_e, T_ion, elec_dens, neutral_dens)
#M
#M Inputs:
#M   T_e           - 1xN array of electron temperatures (K)
#M   T_ion         - 1xN array of ion temperatures (K)
#M   neutral_dens  - MxN array of number density of neutral species, in cm^-3
#  The index below are Matlab index values. for Python subtract 1
#M                     M = 1 : Helium (He)
#M                     M = 2 : Atomic Oxygen (O)
#M                     M = 3 : Nitrogen (N2)
#M                     M = 4 : Oxygen (O2)
#M                     M = 5 : Argon
#M                     M = 6 : Unused
#M                     M = 7 : Atomic Hydrogen (H)
#M                   The routine nrlmsise00 will generate this array.
#M
#M Output:
#M   coll_freq  - the effective electron-neutral collision frequency (Hz)
#M                for the input neutral species
#M
#M   V1.0  M.A. Cervera  10/09/2018
#M     Initial version
#M
#M   V1.1  M.A. Cervera  30/07/2019
#M     Minor update to allow routine to return NaN for collision frequency
#M     if the input electron density is zero
#M
#          W.C. Liles 01/07/2020
#              convert to Python
from Maths import eff_coll_freq_neutrals
from Maths import eff_coll_freq_ion
import numpy as np
# function coll_freq = eff_coll_freq(T_e, T_ion, elec_dens, neutral_dens)
def eff_coll_freq(T_e, T_ion, elec_dens, neutral_dens):
  #M effective collision frequency between electrons and various
  nu_eN2 = eff_coll_freq_neutrals.eff_coll_freq_neutrals(T_e, neutral_dens[2], 0)
  nu_eO2 = eff_coll_freq_neutrals.eff_coll_freq_neutrals(T_e, neutral_dens[3], 1)
  nu_eH  = eff_coll_freq_neutrals.eff_coll_freq_neutrals(T_e, neutral_dens[6], 6)
  nu_eHe = eff_coll_freq_neutrals.eff_coll_freq_neutrals(T_e, neutral_dens[0], 7)
  nu_eO  = eff_coll_freq_neutrals.eff_coll_freq_neutrals(T_e, neutral_dens[1], 8)
  nu_eAr = eff_coll_freq_neutrals.eff_coll_freq_neutrals(T_e, neutral_dens[4], 9)
  
  #M Effective electron-ion collision frequency. 
  nu_ei = eff_coll_freq_ion.coll_freq_ion(T_e, T_ion, elec_dens)

  #M total effective electron collision frequency
  #print(len(nu_ei))
  
  
  #np.nan_to_num(arr)
  arr = np.sum([nu_eN2,nu_eO2, nu_eO, nu_eH, nu_eHe, nu_eAr], axis=0, dtype=object)
  arr.shape = (arr.shape[1])
  aa= np.sum([arr,nu_ei], axis=0, dtype=object)
  
  coll_freq = aa#np.sum(nu_eN2 + nu_eO2 + nu_eO + nu_eH + nu_eHe + nu_eAr + nu_ei)
  #import ipdb; ipdb.set_trace()
  return coll_freq     

