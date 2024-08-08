import matplotlib 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import os
import netCDF4 as nc
import datetime as dt
import ipdb
 

def calculate_scale(data,stddevs=2.,lim='auto'):
    if lim == 'auto':
        mean    = np.nanmean(np.abs(data))
        std     = np.nanstd(np.abs(data))
        lim     = mean + stddevs*std
        scale   = (0,lim)
    else:
        scale   = lim

    ticks   = None
    cmap    = matplotlib.cm.viridis

    return scale,ticks,cmap


 # returns the index of the nearest date that is in the array of dates
def nearest_date_index( dates, target_date):
    # Create a DatetimeIndex from the list of dates
    date_index = pd.DatetimeIndex(dates)
    
    # Calculate the difference between each date in the index and the target_date
    date_difference = abs(date_index - target_date)
    
    # Find the index of the nearest date
    nearest_index = date_difference.argmin()
    return nearest_index

def get_NetCDF_Data(filepath):
        dataset =nc.Dataset(filepath)
        return dataset


def plot_maps(alt=250.,output_dir='output',figsize=(14,8),
            xlim=None,ylim=None,plot_profile_paths='all', target_date:dt.datetime=None, filepath =""
            ,tx_call=None, tx_lon =None, tx_lat=None, rx_call = None, rx_lon = None, rx_lat = None):
        ds = get_NetCDF_Data(filepath)
        lats = np.array(ds["lat0G"][:])
        lons = np.array(ds["lon0G"][:])
        alts = np.array(ds["alt0G"][:])
        time = ds["time"][:]

        start_time =dt.datetime(target_date.year, target_date.month, target_date.day)


        dates = [start_time]
        for x in time:
             if(x!=0):
                next_time = dt.timedelta(hours = float(x))
                dates.append(start_time+next_time - dt.timedelta(days=1))

        dInx = nearest_date_index(dates, target_date)
        edens = ds["dene0G"][:,:,:,:]

        alt_inx = np.argmin(np.abs(alts-alt))
        scale,ticks,cmap = calculate_scale(edens[:,:,alt_inx,:])
        bounds      = np.linspace(scale[0],scale[1],256)
        norm        = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

        this_edens  = edens[dInx,:,:]
        date:dt.datetime = dates[dInx]

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(111,projection=ccrs.PlateCarree())

    #        ax.coastlines(zorder=10,color='k')
    #        ax.add_feature(cartopy.feature.LAND)
    #        ax.add_feature(cartopy.feature.OCEAN)
    #        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    #        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        ax.set_title('')
        ax.gridlines(draw_labels=True)
        LONS, LATS = np.meshgrid(lons,lats)
        data        = this_edens[:,alt_inx,:]
        data =np.moveaxis(data, 0,1)
        pcoll       = ax.pcolormesh(LONS,LATS,data[:-1, :-1],cmap=cmap,norm=norm)

        if plot_profile_paths == 'all':
              
            if tx_lon > 180:
                tx_lon = tx_lon - 360.

            if rx_lon > 180:
                rx_lon = rx_lon - 360.

            ax.scatter(tx_lon,tx_lat,marker='*',s=450,zorder=110,label=tx_call,ec='k',fc='red')
            ax.scatter(rx_lon,rx_lat,marker='*',s=450,zorder=110,label=rx_call,ec='k',fc='orange')
            fontdict = {'size':'x-large','weight':'bold'}
            offset = 1.1
            ax.text(tx_lon,tx_lat+offset,tx_call,fontdict=fontdict,ha='center')
            ax.text(rx_lon,rx_lat+offset,rx_call,fontdict=fontdict,ha='center')
           
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            cbar_label  = r'SAMI3 Electron Density [m$^{-3}$]'
            cbar        = fig.colorbar(pcoll,orientation='vertical',shrink=0.65,pad=0.075,ticks=ticks)
            cbar.set_label(cbar_label,fontdict={'weight':'bold','size':'large'})

            # Plot Title
            txt = []
            txt.append('{0} - Alt: {1:.0f} km'.format(date.strftime('%d %b %Y %H%M UT'),float(alts[alt_inx])))
            ax.set_title('\n'.join(txt),fontdict={'weight':'bold','size':'xx-large'})

            fname = '{0}_{1:03.0f}km_edens_map.png'.format(date.strftime('%Y%m%d_%H%MUT'),float(alts[alt_inx]))

            return ax

            # ################################################################################
            # # Plot electron density profiles of endpoints for validation with IRI runs on CCMC
            # # https://kauai.ccmc.gsfc.nasa.gov/instantrun/iri/
            # if plot_profile_paths == 'all':
            #     for prof_key,profile in self.profiles.items():
            #         fig     = plt.figure(figsize=(15,8))
            #         nrows   = 2
            #         ncols   = 1
            #         ax_inx  = 0
            #         pfxs    = ['tx','rx']
            #         for pfx in pfxs:
            #             ax_inx += 1
            #             ax  = fig.add_subplot(nrows,ncols,ax_inx)

            #             # Get latitude, altitude, and call of endpoint.
            #             call     = profile.attrs['{!s}_call'.format(pfx)]
            #             lat      = profile.attrs['{!s}_lat'.format(pfx)]
            #             lon      = profile.attrs['{!s}_lon'.format(pfx)]
            #             if lon > 180:
            #                 lon = lon - 360.

            #             # Find closest lat/lon in currently calculated electron density array.
            #             clst_lat_inx    = np.argmin(np.abs(lats-lat))
            #             clst_lat        = lats[clst_lat_inx]

            #             clst_lon_inx    = np.argmin(np.abs(lons-lon))
            #             clst_lon        = lons[clst_lon_inx]

            #             edp = edens[dInx,clst_lat_inx,clst_lon_inx,:]
            #             ax.plot(alts,edp,marker='.')
            #             ax.grid(True)

            #             ax.set_xlabel('Altitude [km]')
            #             ax.set_ylabel(r'IRI Electron Density [m$^{-3}$]')
            #             actual      = '({:0.1f}\N{DEGREE SIGN} N, {:0.1f}\N{DEGREE SIGN} E)'.format(clst_lat,clst_lon)
            #             requested   = '{!s} ({:0.1f}\N{DEGREE SIGN} N, {:0.1f}\N{DEGREE SIGN} E)'.format(call,lat,lon)

            #             ax.set_title('Endpoint: {!s}\nActual Plotted: {!s}'.format(requested,actual))
                    
            #         title   = []
            #         title.append('IRI Endpoint Profiles')
            #         title.append('{!s}'.format(date.strftime('%Y %b %d - %H%M UT')))
            #         fig.text(0.5,1.00,'\n'.join(title),ha='center',va='bottom',fontdict={'weight':'bold','size':'large'})

            #         fname = '{!s}_{!s}-{!s}_endPointProfiles'.format(date.strftime('%Y%m%d_%H%MUT'),
            #                 profile.attrs['tx_call'],profile.attrs['rx_call'])
            #         _filename = os.path.join(output_dir,fname)

            #         fig.tight_layout()
            #         fig.savefig(_filename,bbox_inches='tight')
            #         plt.close()