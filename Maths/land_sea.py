#M
#M Name:
#M   land_sea                                
#M
#M Purpose:
#M   Routine to define land or sea globally. Resolution is 0.1 degrees.    
#M                                                                       
#M Calling sequence:
#M   terrain_type = land_sea(glat, glong)
#M
#M Inputs: 
#M   glat  - geographic latitude  (-90 to 90 degrees)             
#M   glong - geographic longitude (0 to 360 degrees measured East from prime
#M           meridien)                            
#M                                                                       
#M Output: 
#M   terrain_type - 0 = sea, 1 = land                             
#M                                                                       
#M Dependencies:
#M   None.
#M
#M Modification history:
#M   27/10/2005  V1.0  M. A. Cervera 
#M     Initial version.
#M
#M   06/05/2010  V1.1  D. J. Netherway  
#M     Allow arguments to be equal sized arrays
#M
#
#    29/07/2020  W. C. Liles
#      Convert to Python
#
import os
import numpy as np
#

# function terrain_type = land_sea(glat, glong)
def land_sea(glat, glong):

    #M Obtain the reference data directory fom the relevant environment 
    # refdata_dir = os.getenv('DIR_MODELS_REF_DAT')
    refdata_dir = "/home/alex/VisualStudioCode/raytrace/pharlap_in_python"
    #M open the data file and read in the land/sea data if this is the first
    #M call to land_sea.m
  
    try:
        land_sea.used += 1
    except AttributeError:
        land_sea.used = 1
        filename = refdata_dir + '/global_land_Mask_3600_by_1800.dat'
      
        with open(filename, 'r') as fid:
              data_str = fid.readline()
    
              # error('#Ms\n#Ms\n#Ms', ...
          	  #  'Missing data file : global_land_Mask_3600_by_1800.dat', ...
          	  #  'Please set the DIR_MODELS_REF_DAT environment variable to point ',...
          	  #  'to the data file directory')
              data_list= list(data_str)
              data_array = np.asarray(data_list)
              data = np.ones(len(data_list),dtype = int)
              data[data_array == '\x00'] = 0
          
              # map_data = fix(fscanf(fid, '%c', [3600, 1800]))
              no_rows = 3600
              no_cols = 1800
              map_data = np.reshape(data,(no_rows, no_cols), order = 'F')
              land_sea.map_data = map_data
              land_sea.data = data
      
    if type(glat) is int:   # assume glong is also int
        glat = np.array([glat])
        glong = np.array([glong])
        
    #M determine whether the lat/long is land or sea.
    zlong = np.mod(glong, 360)
    zlat = glat + 90
    ilat = np.mod(np.round(10*zlat), 1800).astype(int)  #+ 1
    ilong = np.mod(np.round(10*zlong), 3600).astype(int)  #+ 1
    #M terrain_type = map_data(ilong, ilat)

    #M Ensure arrays are one dimensional
    vec_size_long = int(np.prod(ilong.shape))
    ilong = np.reshape(ilong,(1, vec_size_long))[0]
    vec_size_lat = int(np.prod(ilat.shape))
    
    ilat = np.reshape(ilat,(1, vec_size_lat))[0]
    nrows = land_sea.map_data.shape[0]
  
    # terrain_type = NaN(size(ilat))
    
    terrain_type = np.empty(vec_size_lat)


    terrain_type[:] = np.NaN
    valid_terain = np.nonzero(np.invert(np.isnan(ilat)))

    terrain_type[valid_terain] = land_sea.data[ilong[valid_terain] +
                                          (ilat[valid_terain]-1)*nrows]
  
    # terrain_type = np.reshape(terrain_type, np.shape(glat))
  
    return terrain_type


