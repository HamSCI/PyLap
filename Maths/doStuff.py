#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:10:29 2020

@author: william
"""

def doStuff(x):
    try:
        doStuff.timesUsed += 1
    # except AttributeError:
    except:
        doStuff.timesUsed = 1
    # ... special case for first call ...
        print('first time')
        
    # ...common code...
    print(doStuff.timesUsed)
    print('function end')
   
    