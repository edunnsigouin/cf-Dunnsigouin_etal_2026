"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

filename1 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/sipa/sipa_preprocessed_s2s_drammen_2020-01-02_2023-03-13.nc'
filename2 = '/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/sipa/original/sipa_preprocessed_s2s_drammen.nc'

ds1 = xr.open_dataset(filename1)
ds2 = xr.open_dataset(filename2)

tp1 = ds1['tp24']
tp2 = ds2['tp24']

print(ds1)
print(ds2)
print('')
print(ds1['i_date'][-51:].values)
print('')
print(ds2['i_date'][-51:].values)
"""


import xarray as xr
import numpy as np

original = xr.open_dataset(
    "/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/sipa/original/sipa_preprocessed_s2s_drammen.nc"
)

rewritten = xr.open_dataset(
    "/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/sipa/sipa_preprocessed_s2s_drammen_2020-01-02_2023-06-26.nc"
)

print("Same number of i_dates:")
print(
    original.sizes["i_date"]
    ==
    rewritten.sizes["i_date"]
)

print("Same i_date order:")
print(
    np.array_equal(
        original.i_date.values,
        rewritten.i_date.values,
    )
)

print("Same i_dates after sorting:")
print(
    np.array_equal(
        np.sort(original.i_date.values),
        np.sort(rewritten.i_date.values),
    )
)


original_dates = original.i_date.values
rewritten_dates = rewritten.i_date.values

missing = np.setdiff1d(
    original_dates,
    rewritten_dates,
)

print(f"Missing i_dates: {len(missing)}")

for d in missing:
    print(d)
