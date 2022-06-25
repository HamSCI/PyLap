#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:58:29 2020

@author: william
"""

def test_per():
    try:
        print(test_per.flagb)
    except:
        print('tried to print')
    try:
        test_per.flagb += 1
        print('here in try')
    except AttributeError:
        print('here for first')
        test_per.flagb = 1

        
    print('here for next')
    return