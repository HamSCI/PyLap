#!/usr/bin/env python3
#M 
#M  Name :
#M    chapman.m
#M  
#M  Purpose:
#M    Calculates ionospheric plasma frequency profile as a function of height
#M    from input ionospheric layer parameters using Chapman layer profiles.
#M 
#M  Calling sequence:
#M    plasma_freq = chapman(foE, hmE, ymE, foF1, hmF1, ymF1, foF2, ...
#M                          hmF2, ymF2, height) 
#M  
#M  Inputs:
#M    foE  - critical frequency of E layer (scaler)
#M    hmE  - height of E layer (scaler)
#M    ymE  - semi thickness of E layer (scaler)
#M    foF1 - critical frequency of F1 layer (scaler)
#M    hmF1 - height of F1 layer (scaler)
#M    ymF1 - semi thickness of F1 layer (scaler)
#M    foF2 - critical frequency of F2 layer (scaler)
#M    hmF2 - height of F2 layer (scaler)
#M    ymF2 - semi thickness of F2 layer (scaler)
#M    heights - array of heights at which the plasma frequency is to be
#M              calculated
#M 
#M  Outputs:
#M    plasma_freq - array of calculated plasma frequencies
#M  
#M  Modification History:
#M  20/10/2005  V1.0  M.A.Cervera Author.
#M 
#M  07/04/2008  V1.1  M.A.Cervera
#M    The maximum value of each ionospheric layer is now calculated 
#M 
#M  15/04/2008  V1.2  M.A.Cervera
#M    Bug-fix: if maximum value of F1 is -ve then set F1 to zero and
#M    recalculate the maximum value of remaining layers (E and F2)
#M 
#   26/07/2020  W. C. Liles
#     convert to Python
#
import numpy as np
#function plasma_freq = chapman(foE, hmE, ymE, foF1, hmF1, ymF1, foF2, ...
#     hmF2, ymF2, height)  
def chapman(foE, hmE, ymE, foF1, hmF1, ymF1, foF2, hmF2, ymF2, height):

  #M  first calculate the ionospheric Chapman layer maximum values 
  a12 = chap_func(0.5, 2. * (hmE - hmF1) / ymF1) 
  a13 = chap_func(1.0, np.sqrt(2.) * (hmE - hmF2) / ymF2) 
  a21 = chap_func(0.5, 2. * (hmF1 - hmE) / ymE) 
  a23 = chap_func(1.0, np.sqrt(2) * (hmF1 - hmF2) / ymF2) 
  a31 = chap_func(0.5, 2 * (hmF2 - hmE) / ymE) 
  a32 = chap_func(0.5, 2 * (hmF2 - hmF1) / ymF1) 
  A = np.arrya([[1,   a12, a13],
                [a21, 1,   a23],
                [a31, a32, 1]])
  
  b = np.arraya([foE ** 2,  foF1 ** 2,  foF2 ** 2])
  
  lmv_sq = np.linalg.inv(A) * b 
  
  #M  check to see if the square of the F1 layer maximum plasma frequency has a
  #M  solution which is less than 0. This is not physical, in this case repeat,
  #M  but construct the A and b matricies with foF1 set to 0.
  if lmv_sq(2) < 0:
    A[1][0] = 0 
    A[1][2] = 0 
    b[1] = 0 
    lmv_sq = np.inv(A).dot(b) 
  
  lmv_E =  np.sqrt(lmv_sq[0])
  lmv_F1 = np.sqrt(lmv_sq[1]) 
  lmv_F2 = np.sqrt(lmv_sq[2]) 

  #M  plasma frequency contribution from E layer
  plas_E_sq = lmv_E ** 2. * chap_func(0.5, 2 * (height - hmE) / ymE) 
  
  #M  plasma frequency contribution from F1 layer
  plas_F1_sq = lmv_F1 ** 2. * chap_func(0.5, 2 * (height - hmF1) / ymF1) 
  
  #M  plasma frequency contribution from F2 layer
  plas_F2_sq = lmv_F2 ** 2 * chap_func(1, np.sqrt(2.) * (height - hmF2) / ymF2) 
      
  #M  total
  plasma_freq = np.sqrt(plas_E_sq + plas_F1_sq + plas_F2_sq) 
  
  return plasma_freq
 
#M 
#M  define the Chapman function
#M 
#function chap_f = chap_func(C, x)
def chap_func(C,x):
   
   xt = x 
   xt[x > 30] = 30 
   xt[x < -30] = 30 
   
   chap_f = np.exp(C * (1.0 - xt - np.exp(-xt))) 
   
   return chap_f
 