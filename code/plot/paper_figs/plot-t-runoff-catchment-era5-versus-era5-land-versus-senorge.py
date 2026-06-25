import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

filename1 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/senorge/t_gwb_q_1dayacc_regine_drammen_senorge_2023-2023.nc'
filename2 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/era5/t_ro_1dayacc_regine_drammen_era5_0.25x0.25_2023-2023.nc'
filename3 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/era5/t_sro_1dayacc_regine_drammen_era5_0.25x0.25_2023-2023.nc'
filename4 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/era5_land/t_sro_1dayacc_regine_drammen_era5_land_0.1x0.1_2023-2023.nc'

ds_senorge = xr.open_dataset(filename1).sel(time=slice('2023-08-01','2023-08-30'))
ds_era5_ro = xr.open_dataset(filename2)
ds_era5_sro = xr.open_dataset(filename3)
ds_era5_land_sro = xr.open_dataset(filename4).sel(time=slice('2023-08-01','2023-08-30'))


plt.plot(ds_era5_land_sro['time'],ds_era5_land_sro['sro'],'g', label = 'era5_land sro')
plt.plot(ds_senorge['time'],ds_senorge['gwb_q'],'r',label = 'senorge gwb_q')
#plt.plot(ds_era5_ro['ro'],'b', label = 'era5 ro')
#plt.plot(ds_era5_sro['sro'],'k', label = 'era5 sro')


plt.legend()

plt.show()

