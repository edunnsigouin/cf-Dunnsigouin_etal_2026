"""
Calculates an x,y,doy file with the number of ensemble members in all forecast/hindcasts
with x-day accumulated precipitation > storm hans.

The key idea is to look at how the frequency of hans-like events varies by gridpoint so that
we can understand why the retrun period of hans is so dependent on the xy box size and location
in the unseen analysis. 
"""

import numpy
import xarray
from Dunnsigouin_etal_2026 import config, misc

# input ----------------------------------------
date_hans  = '2023-08-07'
accu_days  = 2
write2file = False
# ----------------------------------------------
