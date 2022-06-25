#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
# test program for wgs842gc_lat and earth_radius_wgs84
#"""
import earth_radius_wgs84 as ewgs
import wgs842gc_lat as wgslat
re_eq = 6378137.0                 #M equatorial radius of Earth
origin_lat = -23.5           #M latitude of the start point of ray
    #M convert geodetic origin_lat and origin_lon to lat and lon on a spherical
    #M surface of radius 6378137 m (equatorial radius of Earth) directly above
    #M the origin
re_wgs84 = ewgs.earth_radius_wgs84(origin_lat)
ht = re_eq - re_wgs84
origin_lat_gc = wgslat.wgs842gc_lat(origin_lat, ht)
#origin_lon_gc = origin_lon
print(origin_lat,re_wgs84,origin_lat_gc)

