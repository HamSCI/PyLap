import netCDF4 as nc
import scipy as sp
import numpy as np
import ipdb
import datetime as dt
import pandas as pd
import tqdm
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84

class gen_SAMI3_iono_grid_2d:

    # Constructor to initialize the needed values to do the conversion
    def __init__(self, filepath ,azimuth, transmitter_latitude, transmitter_longitude, range_step, height_step, distance, dateTime,start_time):
        self.filepath = filepath
        self.azimuth = azimuth
        self.transmitter_latitude = transmitter_latitude
        self.transmitter_longitude = transmitter_longitude
        self.range_step = range_step
        self.height_step = height_step
        self.distance = distance
        self.dateTime = dateTime
        self.start_time = start_time

    # calls all of the funtions necesary to create the 2d profile and then returns the 2d profile 
    def get_2d_profile(self):
        SamiData = self.get_NetCDF_Data()
        interpolated_profile = self.generate_2d_profile(SamiData)
        return interpolated_profile
    
    # gets the SAMI3 data from the provided filepath
    def get_NetCDF_Data(self):
        dataset =nc.Dataset(self.filepath)
        
        self.print_NetCDF_Vars(dataset)
        return dataset
    
    # prints out all variables in the netcdf file
    def print_NetCDF_Vars(self,dataset):
        for dim in dataset.dimensions.values():
            print (dim)
        for var in dataset.variables.values():
            print (var)

    # returns the index of the nearest date that is in the array of dates
    def nearest_date_index(self, dates, target_date):
        # Create a DatetimeIndex from the list of dates
        date_index = pd.DatetimeIndex(dates)
    
        # Calculate the difference between each date in the index and the target_date
        date_difference = abs(date_index - target_date)
    
        # Find the index of the nearest date
        nearest_index = date_difference.argmin()

        if(nearest_index ==0):
            return 1
        else:
            return nearest_index

    # a method used to pad the array of data to the ground if the alittudes to not reach the ground 
    def pad_array_to_ground(self, ionosphere, lowest_altitude):
        if(lowest_altitude>0):
            num_zeros:int = int( np.floor(lowest_altitude//self.height_step))
            paded_ionosphere = np.pad(ionosphere, ((0,0),(num_zeros,0)), mode='constant')
            return paded_ionosphere
        else:
            return ionosphere

    # Creates a 2D slice profile of the 3D SAMI3 grid for a transmitter location and a given Azimuth
    def generate_2d_profile(self, SAMI_Data):

        # Setup variables from data
        try:
            lats = np.array(SAMI_Data["lat0G"][:])
            lons = np.array(SAMI_Data["lon0G"][:])
            alts = np.array(SAMI_Data["alt0G"][:])
            time = SAMI_Data["time"][:]

        except:
            lats = np.array(SAMI_Data["lat0"][:])
            lons = np.array(SAMI_Data["lon0"][:])
            alts = np.array(SAMI_Data["alt0"][:])
            time = SAMI_Data["time"][:]    
        # convert to dates format that is usable 
        self.dates = [self.start_time]
        for x in time:
             if(x!=0):
                next_time = dt.timedelta(hours = float(x))
                self.dates.append(self.start_time+next_time - dt.timedelta(days=1))
                
        selected_date_idx = self.nearest_date_index(self.dates,self.dateTime)
        print (selected_date_idx)
        if(selected_date_idx ==0):
            ipdb.set_trace()
        try:
            electron_density = SAMI_Data["dene0G"][selected_date_idx,:,:,:]

        except:
            electron_density = SAMI_Data["dene0"][selected_date_idx,:,:,:]

        # changes longitudes from (0,360) to (-180,180) and sort
        tf = lons > 180
        lons[tf] = lons[tf]-360
       
        # switch altitude to the last index 
        electron_density = np.moveaxis(electron_density,1,-1)

        # switch lattitude and logitude indexes
        electron_density = np.moveaxis(electron_density,0,1)

        # Create meshgrid of original coordinates.
        LATS, LONS, ALTS    = np.meshgrid(lats,lons,alts,indexing='ij')
        coords              = list(zip(LATS.flatten(),LONS.flatten(),ALTS.flatten()))

        # Determine the actual path of the raytrace for a given starting point, distance, azimuth
        Direct_Line = geod.DirectLine(self.transmitter_latitude,self.transmitter_longitude, self.azimuth, self.distance*1000)
        ranges  = np.arange(0,self.distance,self.range_step)

        glats   = []
        glons   = []
        for x in ranges:
            total_distance   = min(x*1e3,Direct_Line.s13) # invl.s13 is the total line distance in m
            tmp = Direct_Line.Position(total_distance,Geodesic.STANDARD)
            glat        = tmp['lat2']
            glon        = tmp['lon2']
            glats.append(glat)
            glons.append(glon)
        flats_flons = np.array([glats,glons]).T
        # Put the field points into a mesh.
        fLATS   = np.zeros([len(ranges),len(alts)])
        fLONS   = np.zeros([len(ranges),len(alts)])
        fALTS   = np.zeros([len(ranges),len(alts)])
        for rInx,flat_flon in enumerate(flats_flons):
            fLATS[rInx,:]   = flat_flon[0]
            fLONS[rInx,:]   = flat_flon[1]
            fALTS[rInx,:]   = alts

        # do the actual interpolation into the 2D slice for a single 
        Ne_profile  = np.zeros([1,len(ranges),len(alts)])
        this_edens          = electron_density[:,:,:]
            
        # interp  = sp.interpolate.LinearNDInterpolator(coords,this_edens.flatten())
        interp  = sp.interpolate.NearestNDInterpolator(coords,this_edens.flatten())

        edens_profile       = interp(fLATS,fLONS,fALTS)
        Ne_profile[:,:]    = edens_profile

        return self.pad_array_to_ground( Ne_profile[0,:,:],alts[0]), self.dates[selected_date_idx]


        # # Create XArray Dataset
        # ds  = xr.Dataset(
        #         data_vars=dict(
        #             electron_density = (['date','range','alt'],Ne_profile),
        #             glats            = (['range'],field_lats),
        #             glons            = (['range'],field_lons),
        #             edensTHT         = (['range'],edensTHT)
        #             ),
        #         coords=dict(
        #             date        = self.dates,
        #             range       = ranges,
        #             alt         = self.alts
        #             ),
        #         attrs=dict(
        #             tx_call     = tx_call,
        #             tx_lat      = tx_lat,
        #             tx_lon      = tx_lon,
        #             rx_call     = rx_call,
        #             rx_lat      = rx_lat,
        #             rx_lon      = rx_lon,
        #             azm         = az,
        #             fname_base  = fname_base
        #         )
        #     )

        # self.profiles[dict_key] = ds
        # return self.profiles


## to do ##
# - make sure we are saving the interpolated data set as a netcdf file. 
# - possibly have to make the change to account for the altitudes being irregular
# - include plotting code for map
   


