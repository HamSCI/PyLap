#!/usr/bin/env python3
#M
#M NAME:
#M	DERIV
#M
#M PURPOSE:
#M	Perform numerical differentiation using 3-point, Lagrangian 
#M	interpolation.
#M
#M CATEGORY:
#M	Numerical analysis.
#M
#M CALLING SEQUENCE:
#M	Dy = Deriv(Y)	 	#MDy(i)/di, point spacing = 1.
#M	Dy = Deriv(Y, X)	#MDy/Dx, unequal point spacing.
#M
#M INPUTS:
#M	Y:  Variable to be differentiated.
#M	X:  Variable to differentiate with respect to.  If omitted, unit 
#M	    spacing for Y (i.e., X(i) = i) is assumed.
#M
#M OPTIONAL INPUT PARAMETERS:
#M	As above.
#M
#M OUTPUTS:
#M	Returns the derivative.
#M
#M SIDE EFFECTS:
#M	None.
#M
#M RESTRICTIONS:
#M	None.
#M
#M PROCEDURE:
#M	See Hildebrand, Introduction to Numerical Analysis, Mc Graw
#M	Hill, 1956.  Page 82.
#M
#M MODIFICATION HISTORY:
#M	Written, DMS, Aug, 1984.
#M       Translated into MATLAB from IDL, M. A. Cervera, HFRD, Jan 1996
#M-
#M
#
#    27/06/2020 W. C. Liles
#      convert to Python
#
import sys
import numpy as np
#
#function d = deriv(y,x)
def deriv(y,*args):
    n = len(y)
    if n < 3: 
        raise ValueError('Parameters must have at least 3 points')
        sys.exit('deriv')
    
    if args:
        x = args[0]
        if n != len(x): 
            print('Vectors must have same size')
            sys.exit('deriv')
        # d = ([y(2:n),y(1)] - [y(n),y(1:n-1)])./([x(2:n),x(1)] - [x(n),x(1:n-1)])
        d_num = np.roll(y, -1) - np.roll(y, 1)
        d_den = np.roll(x, -1) - np.roll(y, 1)
        d = d_num / d_den
        d[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (x[2] - x[0])
        d[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (x[-1] - x[-3])
    else:
        d = (np.roll(y, -1) - np.roll(y, 1)) / 2.0
        d[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / 2.0
        d[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / 2.0
    return d











