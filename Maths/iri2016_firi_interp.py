#M
#M Name :
#M   iri2016_firi_interp.m
#M
#M Purpose :
#M   Matlab wrapper to the IRI-2016 fortan based empirical model ionosphere,
#M   using the FIRI rocketsonde-based model for the region below 120 km, and
#M   interpolating between the FIRI model and the default IRI F layer model.
#M   For (1) a given user specified R12 index (i.e. ionospheric condition), or 
#M   (2) a given epoch, or (3) user specified ionospheric layers, it returns
#M   modeled electron density, electron temperature, ion temperature, and ion
#M   composition (O+, H+, He+, NO+, O+2) height profiles, ion drift and F1
#M   probability along with various other parameters including foE, hmE, foF1,
#M   hmF1, foF2, hmF2. Storms are not modelled for case (1) and may be
#M   switched on or off for case (2). NB: SEE NOTES SECTION BELOW FOR
#M   IMPORTANT INFORMATION.  
#M
#M Calling sequence :
#M    1. [iono, iono_extra] = iri2016_firi_interp(lat, lon, R12, UT, ...
#M                                 ht_start, ht_step, num_hts)
#M
#M    2. [iono, iono_extra] = iri2016_firi_interp(lat, lon, R12, UT, ...
#M                                 ht_start, ht_step ,num_hts, iri_options)
#M
#M Inputs :
#M   lat      - geographic latitude of point (degrees)  
#M   lon      - geographic longitude of point (degrees)
#M   R12      - scalar R12 index
#M      R12 = 1 - 200 :  IRI2016 is called with R12 (Zurich V1.0) input as the
#M                       user specified yearly smoothed monthly median sunspot
#M                       number. The foF2 storm model will be turned off
#M                       regardless of the setting of the optional input
#M                       iri_options.foF2_storm (see below). 
#M      R12 = -1      :  IRI2016 is called with ionospheric conditions (R12,
#M                       IG12, and F10.7) read from file (ig_rz.dat) based on
#M                       input epoch (UT) and may be historical or projected
#M                       conditions (dependent on the epoch). The foF2 storm
#M                       model defaults to "on" for this case (but may be
#M                       overridden). See the optional input
#M                       iri_options.foF2_storm below.   
#M
#M      NB: IRI also requires the IG12 and F10.7 ionospheric parameters to be
#M          supplied. For R12 = -1 these will be read from file (as for R12). 
#M          For a user supplied R12 they are calculated in the mex wrapper
#M          from R12 and input into IRI. The formulae used are:
#M                 F107 = 63.75 + R12 * (0.728 + R12*0.00089)
#M                     (see Davies, "Ionospheric Radio", 1990, pp442)  
#M                 IG12 = -12.349154 + R12 * (1.4683266 - R12*2.67690893e-03)
#M                     (see irisub.for, line 913)
#M
#M   UT       - 5x1 array containing UTC date and time - year, month, day, 
#M              hour, minute
#M   ht_start - start height for ionospheric profile generation (km)
#M   ht_step  - height step size for ionospheric profile generation (km)
#M   num_hts  - number of heights for ionospheric profile generation
#M              must be >= 2 and <= 1000
#M
#M Optional Inputs:
#    Note for the Python version, iri_options is implemented as a dictionary
#     with the keys listed below minus the dot
#M   iri_options - structure containing options to control IRI. If iri_options
#M                 is not used or if a field has not been specfied or has an
#M                 invalid value then the default value is  used. Valid fields
#M                 are : 
#M       .iri_messages - turns IRI messages on or off
#M                         'off' - messages off (default)
#M                         'on'  - messages on
#M
#M       .foF2 - user input foF2. Leave undefined to use IRI model.
#M               Note 1. range of values : 0.1Mz < foE < foF1 < foF2 < 100MHz
#M                    2. If defined then .foF2_storm will be ignored 
#M
#M       .hmF2 - user input hmF2. Leave undefined to use IRI model.
#M               Note range of values : 50km < hmE < hmF1 < hmF2 < 1000km
#M
#M       .foF1 - user input foF2. Leave undefined to use IRI model.
#M               Note range of values : 0.1Mz < foE < foF1 < foF2 < 100MHz
#M
#M       .hmF1 - user input hmF1. Leave undefined to use IRI model.
#M               Note range of values : 50km < hmE < hmF1 < hmF2 < 1000km
#M
#M       .foE  - user input foE. Leave undefined to use IRI model.
#M               Note range of values : 0.1Mz < foE < foF1 < foF2 < 100MHz
#M
#M       .hmE  - user input hmE. Leave undefined to use IRI model.
#M               Note range of values : 50km < hmE < hmF1 < hmF2 < 1000km
#M
#M       .foF2_coeffs - specifies which coefficients to use for foF2 model.
#M                      Valid values are:
#M                        'URSI' - (default)
#M                        'CCIR'
#M
#M       .Ni_model - specifies models to use for ion density profile. Valid
#M                   values are 
#M                     'RBV-2010 & TTS-2005' (default)
#M                     'DS-1995 & DY-1985'
#M
#M       .Te_profile - specifies whether to use standard Te or Te/Ne correlation
#M                  'Te/Ne correlation' - Te calculated using Te/Ne correlation 
#M                  'standard'          - standard Te calculation (default)
#M       
#M       .Te_topside_model - specifies Te topside model
#M                       'TBT-2012' - (default)
#M                       'Bil-1985'
#M
#M       .Te_PF107_dependance - specifies whether to use PF10.7 dependance for 
#M                              the Te model
#M                     'off' - no PF10.7 dependance
#M                     'on'  - with PF10.7 dependance (default)
#M
#M       .Ne_tops_limited - specifies whether or not f10.7 is limited for the
#M                          purpose of calculating the topside electron density
#M                            'f10.7 unlimited' - f10.7 unlimited
#M                            'f10.7 limited'   - f10.7 limited to 188 (default)
#M
#M       .Ne_profile_calc - specifies whether to use standard Ne calculation or
#M                          Ne using Lay-function formalism.
#M                   'Lay-function' - Ne calculated using Lay-function formalism
#M                   'standard'     - standard Ne calculation (default)
#M
#M       .Ne_B0B1_model - string which specifies the model to use for the
#M                     bottomside ionospheric profile. The model defines how
#M                     IRI2016 determines the B0 (thickness) and B1 (shape)
#M                     parameters. The default is ABT-2009. Valid values are:
#M                       'ABT-2009'  (Adv. Space Res., Vol 43, 1825-1834, 2009) 
#M                       'Bil-2000'  (Adv. Space Res., Vol 25, 89-96, 2000)
#M                       'Gul-1987'  (Adv. Space Res., Vol 7, 39-48, 1987)
#M
#M       .Ne_topside_model - string which specifies the model to use for the
#M                           topside ionospheric profile. Valid values are:
#M                      'IRI-2001' - topside model from IRI2001
#M                      'IRI-2001 corrected' - corrected IRI2001 topside model
#M                      'NeQuick' - the NeQuick topside model (default)
#M  
#M       .F1_model - specifies model to use for F1 layer. See Scotto et al., 
#M                   Adv. Space Res., Vol 20, Number 9, 1773-1775, 1997.
#M                     'Scotto-1997 no L' - Scotto without L condition (default)
#M                     'Scotto-1997 with L' - Scotto with L condition
#M                     'solar zenith' - critical solar zenith angle (old IRI95)
#M                     'none' - no F1 layer
#M
#M       .D_model - specifies the model to use for the D Layer. However, the
#M                  input value is overridden and FT-2001 is always used:
#M                  'FT-2001'  - Friedrich and Torkar's FIRI model for the 
#M                               lower ionosphere (Friedrich and Torkar,
#M                               J. Geophys Res. 106 (A10), 21409Ð21418,
#M                               2001). Danilov's et al. D-region model
#M                               (Danilov et al,  Adv. Space Res., vol 15,
#M                               165, 1995) is also returned in the output
#M                               iono(14, :) 
#M
#M       .hmF2_model - specifies the model to use for hmf2. Default is AMTB.
#M                     Valid values are: 
#M                       'AMTB' (default)
#M                       'Shubin-COSMIC'
#M                       'M3000F2'
#M
#M       .foF2_storm - specifies whether or not to have the foF2 storm model on.
#M                     NB: If either R12 has been supplied or foF2 has been user 
#M                     input (i.e. field .foF2 is set) then this will be 
#M                     ignored as it is no longer relevant and the foF2 storm
#M                     model is turned off. 
#M                       'off' - no storm model 
#M                       'on'  - storm model on (default)
#M
#M       .hmF2_storm - specifies whether or not to have foF2 storm model on
#M                     for hmF2 model
#M                       'off' - no storm model (default)
#M                       'on'  - storm model on
#M
#M       .foE_storm - specifies whether or not to have foE storm model on
#M                       'off' - no storm model (default)
#M                       'on'  - storm model on
#M
#M       .topside_storm - specifies whether or not to have foF2 storm model on
#M                        for the topside model
#M                          'off' - no storm model (default)
#M                          'on'  - storm model on
#M       
#M       .auroral_boundary_model - specifies whether to have auroral boundary
#M                                 model on or off
#M                          'off' - auroral boundary model is off (default)
#M                          'on'  - auroral boundary model is on
#M       
#M       .covington - method for calculating Covington Index. Valid values are:
#M                      'F10.7_12' - (default)
#M                      'IG12' - used by IRI before Oct 2015
#M
#M   
#M Outputs :
#M   iono  -  Array of 11 output parameters x num_hts heights. NB, ionosperic
#M            profiles are not computed below 65 km or above 2000 km. Values
#M            of -1 are returned if these heights are requested and also for
#M            those heights where a valid value is unable to be calculated. If
#M            the optional inputs  ht_start and ht_step are not supplied then
#M            the profiles are not calculated.
#      in Python all index values are minus one of listed below
#M       iono(1, :) = electron number density (m^-3)
#M       iono(2, :) = neutral temperature (K)
#M       iono(3, :) = ion temperature (K)
#M       iono(4, :) = electron temperature (K)
#M       iono(5, :) = O+ ion density (m^-3)
#M       iono(6, :) = H+ ion density (m^-3)
#M       iono(7, :) = He+ ion density (m^-3)
#M       iono(8, :) = O2+ ion density (m^-3)
#M       iono(9, :) = NO+ ion density (m^-3)
#M       iono(10, :) = cluster ions density (m^-3)
#M       iono(11, :) = N+ ion density (m^-3)
#M
#M   iono_extra - Array of extra output parameters. NB, Unused array elements  
#M                and parameters which are not able to be calculated are flagged 
#M                by a value of -1.
#      in Python all index values are minus one of listed below
#M      iono_extra(1) = NmF2 (m^-3)         iono_extra(2) = HmF2 (km)
#M      iono_extra(3) = NmF1 (m^-3)         iono_extra(4) = HmF1 (km)
#M      iono_extra(5) = NmE (m^-3)          iono_extra(6) = HmE (km)
#M      iono_extra(7) = NmD (m^-3)          iono_extra(8) = HmD (km)
#M      iono_extra(9) = HHALF (km)          iono_extra(10) = B0 (km)
#M      iono_extra(11) = VALLEY-BASE (m^-3) iono_extra(12) = VALLEY-TOP (km)
#M      iono_extra(13) = Te-PEAK (K)        iono_extra(14) = Te-PEAK HEIGHT (km)
#M      iono_extra(15) = Te-MOD(300KM) (K)  iono_extra(16) = Te-MOD(400KM) (K)
#M      iono_extra(17) = Te-MOD(600KM) (K)  iono_extra(18) = Te-MOD(1400KM) (K)
#M      iono_extra(19) = Te-MOD(3000KM) (K) iono_extra(20) = Te(120KM)=TN=Ti (K)
#M      iono_extra(21) = Ti-MOD(430KM) (K)  iono_extra(22) = X (km), where Te=Ti
#M      iono_extra(23) = sol zen. ang (deg) iono_extra(24) = sun declin. (deg)
#M      iono_extra(25) = DIP (deg)          iono_extra(26) = dip latitude (deg)
#M      iono_extra(27) = modified dip lat.  iono_extra(28) = DELA
#M      iono_extra(29) = sunrise (hours)    iono_extra(30) = sunset (hours)
#M      iono_extra(31) = ISEASON (1=spring) iono_extra(32) = NSEASON (northern)
#M      iono_extra(33) = Rz12               iono_extra(34) = Covington Index
#M      iono_extra(35) = B1                 iono_extra(36) = M(3000)F2
#M      iono_extra(37) = Unused             iono_extra(38) = Unused
#M      iono_extra(39) = gind (IG12)        iono_extra(40) = F1 probability (old)
#M      iono_extra(41) = F10.7 daily        iono_extra(42) = c1 (F1 shape)
#M      iono_extra(43) = daynr              iono_extra(44) = equatorial vertical 
#M      iono_extra(45) = foF2_storm/foF2_quiet               ion drift in m/s
#M      iono_extra(46) = F10.7_81           iono_extra(47) = foE_storm/foE_quiet 
#M      iono_extra(48) = spread-F probability          
#M      iono_extra(49) = Geomag. latitude   iono_extra(50) = Geomag. longitude  
#M      iono_extra(51) = Ap at current time iono_extra(52) = daily Ap
#M      iono_extra(53) = invdip/degree      iono_extra(54) = MLT
#M      iono_extra(55) = CGM-latitude       iono_extra(56) = CGM-longitude
#M      iono_extra(57) = CGM-lati(MLT=0)    iono_extra(58) = CGM-lati for MLT=1
#M      iono_extra(59) = CGM-lati(MLT=2)    iono_extra(60) = CGM-lati for MLT=3
#M      iono_extra(61) = CGM-lati(MLT=4)    iono_extra(62) = CGM-lati for MLT=5
#M      iono_extra(63) = CGM-lati(MLT=6)    iono_extra(64) = CGM-lati for MLT=7
#M      iono_extra(65) = CGM-lati(MLT=8)    iono_extra(66) = CGM-lati for MLT=9
#M      iono_extra(67) = CGM-lati(MLT=10)   iono_extra(68) = CGM-lati for MLT=11
#M      iono_extra(69) = CGM-lati(MLT=12)   iono_extra(70) = CGM-lati for MLT=13
#M      iono_extra(71) = CGM-lati(MLT=14)   iono_extra(72) = CGM-lati for MLT=15
#M      iono_extra(73) = CGM-lati(MLT=16)   iono_extra(74) = CGM-lati for MLT=17
#M      iono_extra(75) = CGM-lati(MLT=18)   iono_extra(76) = CGM-lati for MLT=19
#M      iono_extra(77) = CGM-lati(MLT=20)   iono_extra(78) = CGM-lati for MLT=21
#M      iono_extra(79) = CGM-lati(MLT=22)   iono_extra(80) = CGM-lati for MLT=23
#M      iono_extra(81) = CGM-MLT            iono_extra(82) = CGM-lati for CGM-MLT
#M      iono_extra(83) = Kp at current time iono_extra(84) = magnetic declination
#M      iono_extra(85) = L-value            iono_extra(86) = dipole moment 
#M      iono_extra(87 - 100) = Unused
#M
#M  Notes :
#M   1. Notes for IRI2016 called using specified input ionospheric conditions:
#M   1.1 If the ionospheric conditions are controlled by the matlab input R12
#M       index then the input year (in the UT array) has a very small effect
#M       on  the solar conditions. For example 
#M          [iono iono_extra] = iri2016(-25, 135, 70, [2000 1 1 3 0])
#M       returns NmF2 = 1.0252e+12 electrons/m^3, whereas
#M          [iono iono_extra] = iri2016(-25, 135, 70, [2001 1 1 3 0])
#M       returns NmF2 = 1.0260e+12
#M
#M   1.2 User defined IG12, F10.7 and F10.7_81 (3 solar rotation average of
#M       F10.7) required by IRI2016 are derived from R12 using the following
#M       empirical formulas : 
#M            F107    = 63.7 + 0.728*R12 + 0.00089*R12^2
#M            F107_81 = F107
#M            IG12 = -12.349154 + R12 * (1.4683266 - R12 * 2.67690893e-03)
#M       These derived values for IG12, F10.7 and F10.7_81 are input into 
#M       IRI-2016
#M
#M   2. Notes for IRI2016 called using specified input epoch:
#M   2.1 IRI2016 uses solar indices tabled in ig_rz.dat (which is supplied with
#M       IRI2016). This file contains R12 and IG12 from Jan 1958 to Dec 2018. If 
#M       the input UT is outside this range then an error is returned. R12 is
#M       used to model the height of the F2 layer and IG12 its strength.
#M
#M   2.2 The computation of the yearly-running mean for month M requires the
#M       indices for the six months preceeding M and the six months following 
#M       M (month: M-6, ..., M+6). To calculate the current running mean one 
#M       therefore requires predictions of the index for the next six months. 
#M       Starting from six months before the UPDATE DATE (listed at the top of 
#M       the file ig_rz.dat and below at point 3.) and onward the indices are 
#M       therefore based on indices predictions.
#M
#M   2.3 ig_rz.dat updated 20-Feb-2016
#M
#M   2.4 The solar activity parameter F10.7 (daily) and magnetic activity Ap
#M       index (3-hourly) used by IRI2016 are tabled in apf107.dat from 1 Jan 
#M       1958 to 31 Dec 2014. If UT is outside this range then the storm model
#M       (which relies on  Ap) is switched off and a monthly median F10.7
#M       (calculated from R12 using the empirical formula F107 = 63.7 +
#M       0.728*R12 + 0.00089*R12^2) is used in  place of the daily
#M       F10.7. 
#M
#M   3. This mex file drives IRI-2016 with the following default options
#M      unless over-ridden and input by user via the specification of the 
#M      optional input iri_options.
#M      3.1  Ne computed
#M      3.2  Te, Ti computed
#M      3.3  Ne & Ni computed
#M      3.4  B0,B1 - other models (set by 3.31)
#M      3.5  foF2  - URSI 
#M      3.6  Ni    - RBV-2010 & TTS-2005
#M      3.7  Ne    - f10.7 unlimited
#M      3.8  foF2 from model
#M      3.9  hmF2 from model
#M      3.10 Te    - Standard
#M      3.11 Ne    - Standard Profile
#M      3.12 Messages to unit 6 (but see 3.34 below)
#M      3.13 foF1 from model
#M      3.14 hmF1 from model
#M      3.15 foE  from model
#M      3.16 hmE  from model
#M      3.17 Rz12 from file (unless over-ridden and input by user, i.e. R12 > 0)
#M      3.18 IGRF magnetic field model
#M      3.19 F1 probability model
#M      3.20 standard F1
#M      3.21 ion drift computed
#M      3.22 ion densities in m-3
#M      3.23 Te_topside - TBT-2012
#M      3.24 D-Region model - FT-2001 (can nor be changed by user input)
#M      3.25 F107D from APF107.DAT (unless over-ridden by user, i.e. R12 > 0)
#M      3.26 foF2 with storm model
#M      3.27 IG12 from file (unless over-ridden and input by user, i.e. R12 > 0)
#M      3.28 spread-F probability - computed
#M      3.29 false i.e. topside defined by 3.30 below
#M      3.30 NeQuick topside model 
#M      3.31 B0,B1 model set to ABT-2009
#M      3.32 F10.7_81 from file  (unless over-ridden by user, i.e. R12 > 0)
#M      3.33 Auroral boundary model is off
#M      3.34 Messages off
#M      3.35 no foE storm updating
#M      3.36 hmF2 without foF2-storm                
#M      3.37 topside without foF2-storm 
#M      3.38 turn WRITEs off in IRIFLIP
#M      3.39 hmF2 (M3000F2) false - i.e. new models selected
#M      3.40 hmF2 AMTB model selected
#M      3.41 Use COV=F10.7_12
#M      3.42 Te with PF10.7 dependance
#M
#M Further information :
#M   http://iri.gsfc.nasa.gov/
#M   http://irimodel.org/    
#M
#M Modification History:
#M   V1.0  M. A. Cervera  03/05/2016
#M      Based on V1.3  of iri2012_firi_interp.m
#M
#
#    W. C. Liles 03/07/2020
#       Convert to Python
#
import numpy as np
import scipy.interpolate as sci_int
#

# function [iono, iono_extra] = iri2016_firi_interp(site_lat, site_lon, R12, ...
#      UT, start_height, height_step, num_heights, varargin)
def iri2016_firi_interp(site_lat, site_lon, R12,
      UT, start_height, height_step, num_heights, *varargin):
  
  
  # if (nargin < 7)
  # error('iri2016_firi_interp:argChk', ...
  #	  'Wrong number of input arguments: at leaset 7 inputs required')
  #  return
  # end
  # Python has the first 7 arguments manditory so this error ck not needed 
  iri_options = {}  # create empty dictionary for iri_options
  #M get the iri_options structure if it has been input
  if len(varargin) > 0:
    iri_options = varargin[0]

  
  #M override the input iri_options to make sure FIRI D-layer is selected
  iri_options['D_model'] = 'FT-2001'       
  
  iri2016_arg_str = 'site_lat, site_lon, R12, UT, start_height,' + \
		  'height_step, num_heights, iri_options'
 
  iricall_str = 'iri2016(' + iri2016_arg_str + ')'

  iono, iono_extra = eval(iricall_str)
   
  # height_axis = start_height+(0:(num_heights-1))*height_step
  height_axis = start_height + np.arange(0,num_heights * height_step,
        height_step)
  #M Set limits for interpolation
  bottom_transition = 120        #M always start from 120 km, 
				  #M ignore top 20 km of FIRI model
  if iono_extra[3] > 160:
      #M if F1 present, interpolate to 5 km below F1 peak
      top_transition = iono_extra[3] - 5
  else:
      #M if no F1, interpolate to halfway between 120 km and F2 peak
      top_transition = (bottom_transition + iono_extra[1]) / 2


  if bottom_transition > top_transition:
    bottom_transition = top_transition - 10 
  

  # firi_transition = \
  #    (height_axis > bottom_transition) & (height_axis < top_transition)
  firi_transition = np.logical_and(height_axis > bottom_transition,
                                   height_axis < top_transition)
  # top_idx = find(height_axis > top_transition, 5, 'first')
  top_temp = np.where(height_axis > top_transition)
  if len(top_temp[0]) < 5:
      top_idx = top_temp
  else:
      top_idx = top_temp[0][:5]
  
  # bottom_idx = find(height_axis <= bottom_transition, 5, 'last')
  bottom_temp = np.where(height_axis <= bottom_transition)
  if np.shape(bottom_temp[1]) < 5:
      bottom_idx = bottom_temp
  else:
      bottom_idx = bottom_temp[0][-5:]
      

  # iono(1,firi_transition) = exp(interp1(height_axis([top_idx bottom_idx]), \
  #   log(iono(1,[top_idx bottom_idx])), height_axis(firi_transition), 'pchip'))
  pchip_func = sci_int.PchipInterpolator(
      height_axis[np.concatenate((top_idx, bottom_idx))],
      np.log(iono[0][np.concatenate((top_idx, bottom_idx))]))
  iono[0][firi_transition] = np.exp(pchip_func(height_axis[firi_transition]))
  
      
  valid = iono[0] > 0
  # bottom = find(valid, 5, 'first')
  bottom_temp = np.where(valid)
  if np.shape(bottom_temp)[1] < 5:
      bottom = bottom_temp
  else: 
      bottom = bottom_temp[0][:5]
  iono[0][~valid] = np.exp(np.interp(height_axis(~valid),
                        np.concatenate((np.array([40]), height_axis[bottom])), 
      np.log(np.concatenate((np.array([1e-10]), iono[0][bottom])))))
  iono[0][height_axis < 40] = 0
  return iono, iono_extra
