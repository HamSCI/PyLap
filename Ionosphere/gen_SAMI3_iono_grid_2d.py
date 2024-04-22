import netCDF4 as nc
import scipy as sp
import numpy as np
import ipdb
import datetime as dt
import tqdm
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84

class gen_SAMI3_iono_grid_2d:

    # Constructor to initialize the needed values to do the conversion
    def __init__(self, filepath ,azimuth, transmitter_latitude, transmitter_longitude, range_step, distance, dateTime):
        self.filepath = filepath
        self.azimuth = azimuth
        self.transmitter_latitude = transmitter_latitude
        self.transmitter_longitude = transmitter_longitude
        self.range_step = range_step
        self.distance = distance
        self.dateTime = dateTime

   


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
    def print_NetCDF_Vars(dataset):
        print(dataset.variables)


    # returns the nearest date that is in the array of dates
    def nearest_date(dates, target_date):
        return min(dates, key=lambda x: abs(x - target_date))

    # Creates a 2D slice profile of the 3D SAMI3 grid for a transmitter location and a given Azimuth
    def generate_2d_profile(self, SAMI_Data):

        # Setup variables from data
        lats = np.array(SAMI_Data["lat0G"][:])
        lons = np.array(SAMI_Data["lon0G"][:])
        alts = np.array(SAMI_Data["alt0G"][:])
        time = SAMI_Data["time"][:]
        electron_density = SAMI_Data["dene0G"][:,:,:,:]

        # changes longitudes from (0,360) to (-180,180) and sort
        tf = lons > 180
        lons[tf] = lons[tf]-360
        inx = np.argsort(lons)
        lons = lons[inx]
        electron_density = electron_density[:,inx,:,:]

        # switch altitude to the last index 
        electron_density = np.moveaxis(electron_density,2,-1)

        # switch lattitude and logitude indexes
        electron_density = np.moveaxis(electron_density,1,2)

        # convert to dates format that is usable 
        start_time =dt.datetime(2023,10,13)
        dates = [start_time]
        for x in time:
             if(x!=0):
                next_time = dt.timedelta(hours = float(x))
                self.dates.append(start_time+next_time)


        # Create meshgrid of original coordinates.
        LATS, LONS, ALTS    = np.meshgrid(lats,lons,alts,indexing='ij')
        coords              = list(zip(LATS.flatten(),LONS.flatten(),ALTS.flatten()))

        # Determine the actual path of the raytrace for a given starting point, distance, azimuth
        Direct_Line = geod.DirectLine(self.transmitter_latitude,self.transmitter_longitude, self.azimuth, self.distance*1000)

        # I may need to do the same thing with altitudes so that they are interpolated on a regular grid
        ranges  = np.arange(0,self.distance,self.range_step)

        glats   = []
        glons   = []
        for x in ranges:
            s   = min(x*1e3,Direct_Line.s13) # invl.s13 is the total line distance in m
            tmp = Direct_Line.Position(s,Geodesic.STANDARD)
            glat        = tmp['lat2']
            glon        = tmp['lon2']

            glats.append(glat)
            glons.append(glon)
        flats_flons = np.array([glats,glons]).T

        # Put the field points into a mesh.
        fLATS   = np.zeros([len(ranges),len(self.alts)])
        fLONS   = np.zeros([len(ranges),len(self.alts)])
        fALTS   = np.zeros([len(ranges),len(self.alts)])
        for rInx,flat_flon in enumerate(flats_flons):
            fLATS[rInx,:]   = flat_flon[0]
            fLONS[rInx,:]   = flat_flon[1]
            fALTS[rInx,:]   = self.alts

        # do the actual interpolation into the 2D slice for a single 
            
        selected_date = self.nearest_date(self.dates,self.dateTime)
        Ne_profile  = np.zeros([1,len(ranges),len(self.alts)])
        # for dateInx,date in tqdm.tqdm(enumerate(self.dates),desc='Interpolating Profiles',dynamic_ncols=True):
            # tqdm.tqdm.write('INTERP: {!s}'.format(date))
        this_edens          = electron_density[dateInx,:,:,:]
            
        interp  = sp.interpolate.LinearNDInterpolator(coords,this_edens.flatten())

        edens_profile       = interp(fLATS,fLONS,fALTS)
        Ne_profile[selected_date,:,:]    = edens_profile

        # Calculate range from start as an angle [radians]
        # Computed the same way as in raydarn fortran code.
        field_lats  = flats_flons[:,0]
        field_lons  = flats_flons[:,1]
        edensTHT    = np.arccos( np.cos(field_lats[0]*np.pi/180.)*np.cos(field_lats*np.pi/180.)* \
                            np.cos((field_lons - field_lons[0])*np.pi/180.) \
                    + np.sin(field_lats[0]*np.pi/180.)*np.sin(field_lats*np.pi/180.))

        # Set initial edensTHT = 0
        edensTHT[0] = 0




        return Ne_profile


        # # Create XArray Dataset
        # ds  = xr.Dataset(
        #         data_vars=dict(
        #             electron_density = (['date','range','alt'],Ne_profile),
        #             dip              = (['date','range','alt'],Ne_profile*0),
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



   

  