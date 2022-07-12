#
# Name :
#   pol_power_coupling.m
#
# Purpose :
#   Calculates the amount of power that couples into the O and X propagation 
#   modes
#
# Calling sequence :
#   coupling_factor = pol_power_coupling(axial_ratio_incident, ...
#                              axial_ratio_induced, polariz_angle_change)
#
# Inputs :
#   axial_ratio_incident - the polarization minor to major axial ratio of the
#                          incident radio waves  
#   axial_ratio_induced  - the polarization minor to major axial ratio of the
#                          induced radio waves in the ionosphere
#   polariz_angle_change - the angle between the initial polarization major
#                          axis and the induced polarization major axis 
#
# Outputs :
#   coupling_factor - fraction of power that is coupled into the induced
#                     radio waves in the ionosphere
#
# Notes :
#   See ray_test_3d_pol_coupling.py for an example of how to use this function.
#
# References :
#   Phillips and Knight (1965), "Effects of polarisation on a
#   medium-frequency sky-wave service, including the case of multihop paths",
#   Proc. IEE, Vol. 112, No. 1, pp 31-39
#
# Modification History:
#   V1.0  M.A. Cervera  02/10/2020
#     Initial version.
#
import numpy as np

# function coupling_factor = pol_power_coupling(axial_ratio_incident, 
#       axial_ratio_induced, polariz_angle_change)
def pol_power_coupling(axial_ratio_incident,axial_ratio_induced,polariz_angle_change):

  tmp1 = (1 + axial_ratio_incident * axial_ratio_induced)**2 * \
         np.cos(polariz_angle_change)**2
     
  tmp2 = (axial_ratio_incident + axial_ratio_induced)**2 * \
         np.sin(polariz_angle_change)**2
     
  tmp3 = (1 + axial_ratio_incident**2) * (1 + axial_ratio_induced**2)
  
  coupling_factor = (tmp1 + tmp2) / tmp3
  return coupling_factor


