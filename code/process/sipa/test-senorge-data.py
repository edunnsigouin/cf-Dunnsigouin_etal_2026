import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

filename1 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/sipa/sipa_preprocessed_senorge_drammen_1957-01-01_2022-12-31.nc'
filename2 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/senorge/t_rr_1dayacc_regine_drammen_senorge_1957-2022.nc'

ds1 = xr.open_dataset(filename1).drop_vars('number')
ds2 = xr.open_dataset(filename2).rename({"time": "date"}).rename({"rr":"tp24"})

ds1 = ds1.sel(date=slice("2022-01-01", "2022-12-31"))
ds2 = ds2.sel(date=slice("2022-01-01", "2022-12-31")) 

print(ds1)
print(ds2)
print('')

tp1 = ds1["tp24"].values
tp2 = ds2["tp24"].values

plt.plot(ds1.date,tp1,'k')
plt.plot(ds2.date,tp2-tp1,'b')
plt.show()

