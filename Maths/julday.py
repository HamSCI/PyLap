#M
#M Name :
#M   julday.m
#M 
#M Purpose :
#M   Calculates the Julian Day Number for a given day, month and year.
#M 
#M Calling Sequence :
#M   result = julday(day, month, year)
#M 
#M Inputs:
#M   month -  number of the desired month (1 = january, ..., 12 = december). 
#M   day   -  number of day of the month. 
#M   year  -  number of the desired year.
#M 
#M   all inputs are required to be scalar integers 
#    all inputs are to be numpy arrays of scalar integer values
#    the numpy array might have only one value
#     or  inputs are all integers 
#
#    if inputs are three integers, they are converted to numpy arrays
#M 
#M Outputs:
#M   julday - Julian Day Number (which begins at noon) of the specified
#M            calendar date.
#M 
#M Common blocks:
#M   None.
#M 
#M Dependencies:
#M   None.
#M
#M Restrictions:
#M   Accuracy using IEEE double precision numbers is approximately
#M   1/10000th of a second.
#M 
#M Modification History:
#M   Translated from "Numerical Recipies in C", by William H. Press,
#M   Brian P. Flannery, Saul A. Teukolsky, and William T. Vetterling.
#M   Cambridge University Press, 1988 (second printing).
#
#   Python version keeps to the same date format. Might be changed in the 
#   future to take advantage of Python's date routines
#M 
#M V1.0  Manuel A. Cervera 08/10/1998
#M 
#M V1.1  L.H. Pederick 18/09/2014
#M      Modified to handle vectorized input
#
#      W. C. Liles 29/07/2020
#       Convert to Python
#
#
import numpy as np
#
#function julian = julday(day, month, year)
def julday(day, month, year):
    #M #M make sure that all inputs are scalar
    #M if (length(day) > 1 | length(month) > 1 | length(year) > 1)
    #M   fprintf('\nall inputs are required to be scalar\n')
    #M   return
    #M end
    if type(day) is int:
        day = np.array([day])
        month = np.array([month])
        year = np.array([year])
    
    
    #M gregorian calender was adopted on oct. 15, 1582
    greg = 15 + 31 * (10 + 12 * 1582)
    
    #M process the input
    if any(year == 0):
      print('\nthere is no year zero in julday.py\n')
      return
    
    
    year[year < 0] = year[year < 0] + 1
    
    jy = np.zeros(day.size)
    jm = np.zeros(day.size)
    afterfeb = month > 2
    
    jy[afterfeb] = year[afterfeb]
    jm[afterfeb] = month[afterfeb] + 1
    afterfeb_not = np.invert(afterfeb)
    jy[afterfeb_not] = year[afterfeb_not] - 1
    jm[afterfeb_not] = month[afterfeb_not] + 13
    
    julian = np.floor(365.25 * jy) + np.floor(30.6001 * jm) + day + 1720995
    
    #M test whether to change to gregorian calendar.
    aftergreg = ((day + 31 * (month + 12 * year)) >= greg)
    ja = (0.01 * jy[aftergreg]).astype(int)
    julian[aftergreg] = julian[aftergreg] + 2 - ja + (0.25 * ja).astype(int)
    julian_int = julian.astype(int)
    return julian_int
