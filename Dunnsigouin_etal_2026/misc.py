"""
Collection of useful miscellaneous functions
"""

import time
import numpy  as np
import xarray as xr

def tic():
    """
    matlab style tic function
    """
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    return

def toc():
    """
    matlab style toc function
    """                        
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
    return


def xy_mean(ds):
    """ 
    calculates xy mean over dims lat and lon
    with cosine weighting in lat. Input is xarray
    dataarray or dataset
    """
    weights = np.cos(np.deg2rad(ds.latitude))
    ds      = ds.weighted(weights).mean(dim=('latitude','longitude'))
    return ds        


def rm_lpyr_days(data):
    """ 
    removes leap-year days from daily xrray dataset
    """
    return data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))


def is_leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False

    
def get_domain_latlon(domain):
    if domain == 'norway':
        domain_lats = slice(72,57)
        domain_lons = slice(4,32)
    return domain_lats, domain_lons
