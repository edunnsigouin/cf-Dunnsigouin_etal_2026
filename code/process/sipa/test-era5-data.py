import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

filename1 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/sipa/sipa_preprocessed_era5_drammen_2000-01-01_2023-08-10.nc'
filename2 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/sipa/original/sipa_preprocessed_era5_drammen.nc'
filename3 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/era5/t_tp24_1dayacc_regine_drammen_era5_0.5x0.5_1958-2023.nc'

ds1 = xr.open_dataset(filename1)
ds2 = xr.open_dataset(filename2).sortby("date")
ds3 = (
    xr.open_dataset(filename3)
    .rename({"time": "date"})
    .sel(date=slice("2000-01-01", "2023-08-10"))
)

# ------------------------------------------------------------------
# Extract finite values
# ------------------------------------------------------------------

tp1 = ds1["tp24"].values
#tp1 = tp1[np.isfinite(tp1)]

tp2 = ds2["tp24"].values
#tp2 = tp2[np.isfinite(tp2)]

tp3 = ds3["tp24"].values
#tp3 = tp3[np.isfinite(tp3)]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True,
    sharey=True,
)

axes[0].plot(ds1.date,tp1-tp3,'k')
axes[1].plot(ds1.date,tp2-tp1,'k')
axes[2].plot(ds1.date,tp2-tp3,'k')
plt.show()
