#
# Name:
#   QP_profile_multi_seg.m
#
# Purpose:
#   Returns quasi-parabolic segment coefficients and the electron density
#   profile constructed from those coefficients. The QP segments represent:
#     1. QP ionospheric layers (E, F1 and F2)
#     2. reverse QP joining layers (E to F1 and F1 to F2)
#     3. small reverse QP layer at the  bottom of the profile to give a
#        smooth increasing electron density from zero.
#   The joining segments are gradient matched to the layer segments. See 
#   Dyson and Bennett, 1988 for details.
#
# Calling Sequence:
#    [elec_dens, dN, QP_seg_coeffs] = QP_profile(foE, hmE, ymE, foF1, ...
#           hmF1, ymF1, foF2, hmF2, ymF2, heights, re) 
#
# Inputs:
#   foE       -   maximum plasma frequency of the E layer (MHz)
#   hmE       -   height of maximum electron density of E layer (km)
#   ymE       -   QP semi-thickness of the E layer
#   foF1      -   maximum plasma frequency of the E layer (MHz)
#   hmF1      -   height of maximum electron density of E layer (km)
#   ymF1      -   QP semi-thickness of the E layer
#   foF2      -   maximum plasma frequency of the E layer (MHz)
#   hmF2      -   height of maximum electron density of E layer (km)
#   ymF2      -   QP semi-thickness of the E layer
#   height    -   array of heights at which QP profile is to be calculated (km)
#   re        -   radius of the Earth (approximated as spherical) (km)
#
# Outputs:
#   elec_dens     -  electron density profile (electrons / cm^3)
#   dN            -  derivative of electron density wrt height
#   QP_seg_coeffs -  the coefficients of the quasi parabolic segments (both
#                    actual layers and the joining layers) which define the
#                    electron density profile
#     QPcoeffs(ii, 1) = a     where a = Nm 
#     QPcoeffs(ii, 2) = b     where b = Nm .* ((hm + re - ym) / ym).^2
#     QPcoeffs(ii, 3) = rb    radial coordinate of bottom of segment
#     QPcoeffs(ii, 4) = rt    radial coordinate of top of segment
#     QPcoeffs(ii, 5) = 1 for ionospheric layer (E, F1 or F2)
#                      -1 for joining layer (bottom to E, E to F1, or F1 to F2)
#
# Dependencies:
#   None
#
# References:
#   Dyson and Bennett, J. Atmos. Terr. Phys., Vol. 50, 251-262, 1988
#   Croft and Hoogasian, Radio Science, Vol. 3, 69--74, 1968
#   
# Modification History
#   M. A. Cervera  30/08/2016  V1.0
#      Initial version.
#
import numpy as np  # py


def QP_profile_multi_seg(foE, hmE, ymE, foF1, hmF1, ymF1, foF2, hmF2, 
                          ymF2, heights, re):
 
  plas_fac =  80.6163849431291e-6   # converts plasma freq to electrons / cm^3
  
  r = heights + re 
  
  NmE  = foE**2 / plas_fac
  NmF1 = foF1**2 / plas_fac
  NmF2 = foF2**2 / plas_fac
  rmE  = hmE + re
  rbE  = rmE - ymE  
  rmF1 = hmF1 + re
  rbF1 = rmF1 - ymF1  
  rmF2 = hmF2 + re
  rbF2 = rmF2 - ymF2  
  QP_seg_coeffs = np.ones((6,5))

  # 
  # F2 layer segments
  #

  # calculate the electron density profile and its derivative for the F2 layer
  a = NmF2
  b = NmF2 * (rbF2 / ymF2)**2
  N_F2_layer = a - b*(1 - rmF2/r)**2
  dN_F2_layer = -2*b * (1 - rmF2/r) * rmF2 / r**2
  idx = np.argwhere(N_F2_layer < 0)
  N_F2_layer[idx] = 0
  dN_F2_layer[idx] = 0
    
  # calculate the inverse QP electron density profile and its derivative for
  # the segment which joins the bottom of F2 layer to the top of F1 layer
  aj = NmF1
  rj = rmF1
  rc = (rmF2 * b * (rmF2/rj - 1)) / (a - aj + b*(rmF2/rj - 1))
  bj = -rmF2 * b * (1 - rmF2/rc) / (rj * (1 - rj/rc))
  N_F2_F1_join = aj + bj*(1 - rj/r)**2
  dN_F2_F1_join = 2*bj * (1 - rj/r) * rj / r**2
  idx = np.argwhere(heights+re < rj )
  N_F2_F1_join[idx] = 0
  dN_F2_F1_join[idx] = 0
  
  # populate the QP segment coefficeints array for the F2 layer segment and
  # the F2 - F1 joining segment
  segment = 5
  QP_seg_coeffs[segment][0] = a
  QP_seg_coeffs[segment][1] = b
  QP_seg_coeffs[segment][2] = rc
  QP_seg_coeffs[segment][3] = rmF2
  QP_seg_coeffs[segment][4] = 1   # F2 layer segment

  segment = 4
  QP_seg_coeffs[segment][0] = aj
  QP_seg_coeffs[segment][1] = -bj
  QP_seg_coeffs[segment][2] = rmF1
  QP_seg_coeffs[segment][3] = rc
  QP_seg_coeffs[segment][4] = -1   # joining segment - bottom of F2 to F1
  
  
  # 
  # F1 layer segments
  #

  # calculate the electron density profile and its derivative for the F1 layer
  a = NmF1
  b = NmF1 * (rbF1 / ymF1)**2
  N_F1_layer = a - b*(1 - rmF1/r)**2
  dN_F1_layer = -2.*b * (1 - rmF1/r) * rmF1 / r**2
  idx = np.argwhere(N_F1_layer < 0)
  N_F1_layer[idx] = 0
  dN_F1_layer[idx] = 0
    
  # calculate the inverse QP electron density profile and its derivative for
  # the segment which joins the bottom of F1 layer to the top of E layer
  aj = NmE
  rj = rmE
  rc = (rmF1 * b * (rmF1/rj - 1)) / (a - aj + b*(rmF1/rj - 1))
  bj = -rmF1 * b * (1 - rmF1/rc) / (rj * (1 - rj/rc))
  N_F1_E_join = aj + bj*(1 - rj/r)**2
  dN_F1_E_join = 2*bj * (1 - rj/r) * rj / r**2
  idx = np.argwhere(heights+re < rj )
  N_F1_E_join[idx] = 0
  dN_F1_E_join[idx] = 0
  
  # populate the QP segment coefficeints array for the F1 layer segment and
  # the F1 - E joining segment
  segment = 3
  QP_seg_coeffs[segment][0] = a
  QP_seg_coeffs[segment][1] = b
  QP_seg_coeffs[segment][2] = rc
  QP_seg_coeffs[segment][3] = rmF2
  QP_seg_coeffs[segment][4] = 1   # F1 layer segment

  segment = 2
  QP_seg_coeffs[segment][0] = aj
  QP_seg_coeffs[segment][1] = -bj
  QP_seg_coeffs[segment][2] = rmF1
  QP_seg_coeffs[segment][3] = rc
  QP_seg_coeffs[segment][4] = -1   # joining segment - bottom of F1 to E
  
  
  # 
  # E layer segments
  #
  
  # calculate the electron density profile and its derivative for the E layer
  a = NmE
  b = NmE * (rbE / ymE)**2
  N_E_layer = a - b*(1 - rmE/r)**2
  dN_E_layer = -2*b * (1 - rmE/r) * rmE / r**2
  idx = np.argwhere(N_E_layer < 0)
  N_E_layer[idx]= 0
  dN_E_layer[idx] = 0
    
  # calculate the inverse QP electron density profile and its derivative for
  # the joining segment which smooths the bottom of E layer
  aj = 0.0
  rj = rmE - 1.2*ymE
  rc = (rmE * b * (rmE/rj - 1)) / (a - aj + b*(rmE/rj - 1))
  bj = -rmE * b * (1 - rmE/rc) / (rj * (1 - rj/rc))
  N_E_smooth = aj + bj*(1 - rj/r)**2
  dN_E_smooth = 2*bj * (1 - rj/r) * rj / r**2
  idx = np.argwhere(heights+re < rj )
  N_E_smooth[idx] = 0
  dN_E_smooth[idx] = 0
  
  # populate the QP segment coefficeints array for the E layer segment and
  # the smoothing segment
  segment = 1
  QP_seg_coeffs[segment][0] = a
  QP_seg_coeffs[segment][1] = b
  QP_seg_coeffs[segment][2] = rc
  QP_seg_coeffs[segment][3] = rmF2
  QP_seg_coeffs[segment][4] = 1   # E layer segment

  segment = 0
  QP_seg_coeffs[segment][0] = aj
  QP_seg_coeffs[segment][1] = -bj
  QP_seg_coeffs[segment][2] = rmF1
  QP_seg_coeffs[segment][3] = rc
  QP_seg_coeffs[segment][4] = -1   # joining segment - smooths bottom of E

  # print(heights+re)
  #
  # now construct the electron density frequency profile and its derivative 
  # from the segments
  #
  N_tot = np.zeros(np.size(heights))
  dN_tot = np.zeros(np.size(heights))

  # E smoothing segment
  r_E_join =  QP_seg_coeffs[0][3]
  idx = np.argwhere(heights+re < r_E_join)
  # print(idx)
  N_tot[idx] = N_E_smooth[idx]
  dN_tot[idx] = dN_E_smooth[idx]
  
  # E layer segment
  a = np.argwhere(heights+re >= r_E_join) 
  b = np.argwhere(heights+re < rmE) 
  idx = np.argwhere(b >= a[0][0])
  # print(idx, r_E_join, rmE)
  N_tot[idx] = N_E_layer[idx]  
  dN_tot[idx] = dN_E_layer[idx] 
  
  # F1 - E joining segment
  r_F1_join =  QP_seg_coeffs[2][3]
  a = np.argwhere(heights+re >= rmE) 
  b = np.argwhere(heights+re < r_F1_join) 
  idx = np.argwhere(b >= a[0][0])
  # print(idx, rmE, r_F1_join)
  N_tot[idx] = N_F1_E_join[idx]
  dN_tot[idx] = dN_F1_E_join[idx]

  # F1 layer segment
  a = np.argwhere(heights+re >= r_F1_join) 
  b = np.argwhere(heights+re < rmF1) 
  idx = np.argwhere(b >= a[0][0])
  # print(idx, r_F1_join, rmF1)
  N_tot[idx] = N_F1_layer[idx]
  dN_tot[idx] = dN_F1_layer[idx]
  
  # F2 - F1 joining segment
  r_F2_join =  QP_seg_coeffs[4][3]
  a = np.argwhere(heights+re >= rmF1) 
  b = np.argwhere(heights+re < r_F2_join) 
  idx = np.argwhere(b >= a[0][0])
  # print(idx, rmF1, r_F2_join)
  N_tot[idx] = N_F2_F1_join[idx]
  dN_tot[idx] = dN_F2_F1_join[idx]

  # F2 layer segment
  idx = np.argwhere(heights+re >= r_F2_join)
  # print(idx)
  N_tot[idx] = N_F2_layer[idx]
  dN_tot[idx] = dN_F2_layer[idx]
  
  elec_dens = N_tot
  dN = dN_tot
  # print(QP_seg_coeffs)
  return [elec_dens, dN, QP_seg_coeffs] 
