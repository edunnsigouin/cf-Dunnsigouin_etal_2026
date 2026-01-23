"""
draft figure 01 for hans paper showing obs and hans
"""

import numpy as np
import xarray as xr
from Dunnsigouin_etal_2026 import config, misc


# import ---------------------------------------
path_in                    = config.dirs['obs']
path_out                   = config.dirs['fig']
filename_in_streamflow     = f'{path_in}streamflow.Bergheim.nc'
filename_in_precipitation  = f'{path_in}precipitation.tunhovd.nc'
filename_in_snowdepth      = f'{path_in}snowdepth.tunhovd.nc'
filename_out               = f'{path_out}fig-01.pdf' 
write2file                 = False
# ----------------------------------------------


