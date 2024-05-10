#!/usr/bin/env python3

import numpy as np  # py
import time
import ctypes as c
from pylap.raytrace_2d import raytrace_2d 
from Ionosphere import gen_iono_grid_2d as gen_iono_IRI
from Ionosphere import gen_SAMI3_iono_grid_2d as gen_iono
from Plotting import plot_ray_iono_slice as plot_iono
from Plotting import Plot_map 
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pnd
import ipdb
import os
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84
plt.switch_backend('tkagg')
R12 = 100                    #M R12 index

df = pnd.read_csv("Examples/run_raytraces.csv")
pfsq_conv = 80.6163849431291e-12  #M mult. factor to convert elec. density


# loop through the csv file to run every single ray trace in the file
for index, row in df.iterrows():

    # create the result directories
    tx_label = row["tx_lbl"]
    rx_label = row["rx_lbl"]
    date_utc = row["date_UTC"]
    frequency_MHz = row["freq_MHz"]

    folder_directory = 'Examples/results/{}_{}_{}_{}MHz'.format(tx_label, rx_label, date_utc, frequency_MHz)

    if(os.path.isdir(folder_directory)!=True):
        os.mkdir(folder_directory)

    # parameters to make the ionosphere
    ray_bear = row["tx_end_azm"]            #M bearing of rays
    origin_lat = row["tx_lat"]          #M latitude of the start point of ray
    origin_long = row["tx_lon"]        #M longitude of the start point of ray 

    range_step = 10
    height_step = 5
    Distance = 3000
    datetime = dt.datetime.strptime(date_utc, '%Y-%m-%d %H:%M:%S')
    start_time =dt.datetime(datetime.year, datetime.month, datetime.day)


    filepath = row["filepath"]
    print(filepath)
    ionosphere= gen_iono.gen_SAMI3_iono_grid_2d(filepath,
                                             ray_bear,
                                             origin_lat,
                                             origin_long,
                                             range_step, 
                                             height_step,
                                             Distance, 
                                             datetime,
                                             start_time
                                             )
    # generate 2D slice of ionosphere
    ed,actual_date =ionosphere.get_2d_profile()
    electron_density = np.transpose(ed)
    iono_en_grid = electron_density

    #set all other inputs to zero
    iono_en_grid_5 = np.zeros(electron_density.shape)
    collision_freq = np.zeros(electron_density.shape)
    irreg = np.zeros((4, electron_density[0,:].size))



    elevs = np.arange(1, 80, 2, dtype = float) # py
    num_elevs = len(elevs)
    freq = frequency_MHz                #M ray frequency (MHz)
    freqs = freq * np.ones(num_elevs, dtype = float) # py
    tol = [1e-7, 0.01, 10] 
    nhops = 1                    #M number of hops to raytrace
    start_height =0

    # run ray traces

    ray_data, ray_path_data, ray_path_state = \
    raytrace_2d(origin_lat, origin_long, elevs, ray_bear, freqs, nhops,
               tol, 0, iono_en_grid, iono_en_grid_5,
 	       collision_freq, start_height, height_step, range_step, irreg)
    

    iono_pf_grid = np.sqrt(electron_density * pfsq_conv)*1000
    # here a Subset of the data is specified to be plotted 
    start_range = 0
    start_range_idx = 0
    end_range = Distance -100
    end_range_idx = int(np.round((end_range) / range_step) + 1)
    start_ht = start_height
    start_ht_idx = 0
    end_ht = 500
    end_ht_idx = int((end_ht - start_height) / height_step) + 1
    iono_pf_subgrid = iono_pf_grid[start_ht_idx:end_ht_idx,start_range_idx:end_range_idx]

    ax1, ray_handle = plot_iono.plot_ray_iono_slice(iono_pf_subgrid, start_range,
                        end_range, range_step, start_ht, end_ht, height_step,
                        ray_path_data,linewidth=1.5, color='1')

    fig_str_a = ' SAMI3  {} UTC  {}MHz   '.format(actual_date.replace(microsecond=0), freq)
    fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
                round(origin_lat,2), round(origin_long,2), round(ray_bear, 2))

    fig_str = fig_str_a + fig_str_b




    ax1.set_title(fig_str)


    x = row["tx_rx_range_km"]

    Re=6371
    rx_lat      = row["rx_lat"]
    rx_lon      = row["rx_lon"]

    # Determine the ranges and azimuth along the profile path.
    invl    = geod.InverseLine(origin_lat,origin_long,rx_lat,rx_lon)
    rx_dist_km = invl.s13*1e-3   # Distance in km
    rx_az      = invl.azi1

   

    rx_theta    = (rx_dist_km - start_range - (end_range - start_range)/2)/Re
    rx_x = Re * np.sin(rx_theta)
    rx_y = Re * np.cos(rx_theta)

    # ax1.scatter([0],[6371],s=500,marker='*',color='red',ec='k',zorder=100,clip_on=False,label=rx_label)
    # rx_theta =chu_km/Re
    ax1.scatter([rx_x],[rx_y],s=500,marker='*',color='red',ec='k',zorder=100,clip_on=False,label=rx_label)
    # ax1.legend([hndl],[rx_label],loc='upper right',scatterpoints=1,fontsize='large',labelcolor='black')

    filename = '{}/Sami_raytrace.png'.format(folder_directory)
    if(os.path.isfile(filename)):
        os.remove(filename)

    plt.savefig(filename)



    if(actual_date.year > 2023):
        UT = [2020, actual_date.month, actual_date.day, actual_date.hour, actual_date.minute]#2001,3,15,14,15 makes 7:00UT    #M UT - year, month, day, hour, minute
    else:
        UT = [actual_date.year, actual_date.month, actual_date.day, actual_date.hour, actual_date.minute]#2001,3,15,14,15 makes 7:00UT    #M UT - year, month, day, hour, minute
    #  The above is not a standare Python datetime object but leaving unchanged
    
    doppler_flag = 0             #M generate ionosphere 5 minutes later so that
                                #M Doppler shift can be calculated
    irregs_flag = 0              #M no irregularities - not interested in
                                #M Doppler spread or field aligned irregularities
    kp = 0                       #M kp not used as irregs_flag = 0. Set it to a
                                #M dummy value


    print('\n'
        'Example of 2D numerical raytracing for a fan of rays for a WGS84'
        ' ellipsoidal Earth\n\n') # py

    #M
    #M generate ionospheric, geomagnetic and irregularity grids
    #M
    max_range = 10000       #M maximum range for sampling the ionosphere (km)
    num_range = int(Distance /range_step)        #M number of ranges (must be < 2000)
    # range_inc = max_range / (num_range - 1) # py
    start_height = 0        #M start height for ionospheric grid (km)
    # height_inc = 3          #M height increment (km)
    num_heights = int(500/ height_step)    #M number of  heights (must be < 2000)
    #iri_options.Ne_B0B1_model = 'Bil-2000'  #M this is a non-standard setting for
                                            #M IRI but is used as an example
    # implement the above by means of dictionay
    iri_options = {
                'Ne_B0B1_model': 'Bil-2000'
                }   
    print('Generating ionospheric grid... ')
    iono_pf_grid, iono_pf_grid_5, collision_freq, irreg, iono_te_grid = \
        gen_iono_IRI.gen_iono_grid_2d(origin_lat, origin_long, R12, UT, ray_bear,
                max_range, num_range, range_step, start_height,
                height_step, num_heights, kp, doppler_flag, 'iri2016',
                iri_options)


    #M convert plasma frequency grid to  electron density in electrons/cm^3
    iono_en_grid = (iono_pf_grid ** 2) / 80.6164e-6
    iono_en_grid_5 = (iono_pf_grid_5 ** 2) / 80.6164e-6




    #M
    #M Example 1 - Fan of rays, 10 MHz, single hop. Print to encapsulated
    #M postscript and PNG. Note the transition from E-low to E-High to F2-low modes.
    #M

    #M call raytrace for a fan of rays
    #M first call to raytrace so pass in the ionospheric and geomagnetic grids
    print('Generating {} 2D NRT rays ...'.format(num_elevs))

    ray_data, ray_path_data, ray_path_state = \
    raytrace_2d(origin_lat, origin_long, elevs, ray_bear, freqs, nhops,
                tol, irregs_flag, iono_en_grid, iono_en_grid_5,
            collision_freq, start_height, height_step, range_step, irreg)


    ################
    ### Figure 1 ###
    ################
    start_range = 0
    start_range_idx = int(start_range/range_step) 
    end_range = 2900
    end_range_idx = int((end_range) / range_step) + 1
    start_ht = start_height
    start_ht_idx = 0
    end_ht = 400
    end_ht_idx = int(end_ht / height_step) + 1
    iono_pf_subgrid = iono_pf_grid[start_ht_idx:end_ht_idx,start_range_idx:end_range_idx]


    ax2, ray_handle = plot_iono.plot_ray_iono_slice(iono_pf_subgrid, start_range,
                        end_range, range_step, start_ht, end_ht, height_step,
                        ray_path_data,linewidth=1.5, color=[1, 1, 0.99])


    fig_str_a = 'IRI {}/{}/{}  {:02d}:{:02d}UT   {}MHz   R12 = {}'.format(
                UT[1], UT[2], UT[0], UT[3], UT[4], freq, R12)
    fig_str_b = '   lat = {}, lon = {}, bearing = {}'.format(
                origin_lat, origin_long, ray_bear)

    fig_str = fig_str_a + fig_str_b

    
        
    hndl    = ax2.scatter([rx_x],[rx_y],s=500,marker='*',color='red',ec='k',zorder=100,clip_on=False,label=rx_label)
    # ax2.legend([hndl],[rx_label],loc='upper right',scatterpoints=1,fontsize='large',labelcolor='black')

    ax2.set_title(fig_str)
    filename1 = '{}/IRI_raytrace.png'.format(folder_directory)

    plt.savefig(filename1)

    # ax = Plot_map.plot_maps(target_date=actual_date, filepath=filepath, tx_call=tx_label, rx_call=rx_label,
    #                          tx_lon=origin_long, tx_lat=origin_lat, rx_lat=row["rx_lat"], rx_lon=row["rx_lon"])
    
    # filename2 = '{}/Edens_Map.png'.format(folder_directory)
    # ipdb.set_trace()


    # plt.tight_layout()
    # plt.savefig(filename2,bbox_inches='tight')
    # plt.close()
    


