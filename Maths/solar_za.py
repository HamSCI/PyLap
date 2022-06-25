#M
#M Name :
#M   solar_za.m
#M
#M Purpose :
#M   Calculate the solar zenith angle (apprximate formula).
#M
#M Calling sequence :
#M   solar_zen_ang = solar_za(lat, lon, UT)
#M
#M Inputs :
#M   lat - latitude (degrees)
#M   lon - longitude (degrees)
#M   UT  - 5xN array containing UTC date and time - year,month,day,hour,minute
#M
#M Outputs :
#M   solar_zen_ang - solar zenith angle, degrees as nparray
#M
#M Author:
#M   V1.0  M.A. Cervera  07/08/2009
#M
#M   V1.1  L.H. Pederick 18/09/2014
#M      Modified to handle vectorized input for UT
#
#    W. C. Liles 06/-8/2020
#       convert to Python
#       date is a n x 5 list. am not using Python datetime formats
#       need to detect if only one date as a simple list
#
import numpy as np
from Maths import julday
#

# function solar_zen_ang = solar_za(lat, lon, UT)
def solar_za(lat, lon, UT): 
  

  #M convert lat, lon to radians
  lat_r = np.radians(lat)
  lon_r = np.radians(lon)
  UT_temp = np.asarray(UT)
  if UT_temp.ndim == 1:
      UT_temp = np.array([UT_temp])
  UT_array = np.transpose(UT_temp)
  #M calculate the hour angle, hour_ang
  hour = UT_array[3] + UT_array[4] / 60.                   #M UTC decimal hour
  hour_LST = np.mod(hour + lon / 15, 24)         #M Local Solar Time decimal hour 
  hour_ang = np.radians(15 * (hour_LST - 12))   #M Hour angle
  #M calculate the day number in the year, doy
  doy = julday.julday(UT_array[2], UT_array[1], UT_array[0]) -\
               julday.julday(np.zeros(len(UT_array[2])),
                      np.ones(len(UT_array[2])), UT_array[0])
  
  #M calculate the solar declination, solar_dec (approximate formula)
  obliq_ecliptic = 23.45
  solar_dec = np.radians(obliq_ecliptic * np.sin(2.0 * np.pi *
                                                 (doy + 284) / 365)) 
  
  #M calculate the solar zenith angle in degrees
  csza = np.sin(lat_r) * np.sin(solar_dec) + \
         np.cos(lat_r) * np.cos(solar_dec) * np.cos(hour_ang)
  solar_zen_ang = np.degrees(np.arccos(csza))

  return solar_zen_ang


