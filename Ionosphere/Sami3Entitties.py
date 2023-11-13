#!/usr/bin/env python3

#going to define strongly typed types for each param
class SAMI3Params:
  def __init__(self,origin_lat, origin_lon, R12, UT, azim, 
            max_range, num_range, range_inc, start_height,
            height_inc, num_heights, kp, doppler_flag,filePath,
             *args):
    self.origin_lat = origin_lat
    self.origin_lon = origin_lon
    self.R12 =R12
    self.UT=UT
    self.azim = azim
    self.max_range = max_range
    self.num_range = num_range
    self.rang_inc = range_inc
    self.start_height =start_height
    self.height_inc = height_inc
    self.num_heights = num_heights
    self.kp =kp
    self.doppler_flag = doppler_flag
    self.filePath =filePath
    self.args= args

    ###TODO###
    ###implement what the object will return as a string for easy testing##
    # def __str__(self):
    # return f"{self.name}({self.age})"  