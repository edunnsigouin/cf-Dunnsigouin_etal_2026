"""
hard coded paths in cf-Dunnsigouin_etal_2026
"""

cf_space              = "/nird/datapeak/NS9873K/etdu/"
proj                  = "/nird/home/edu061/cf-Dunnsigouin_etal_2026/"
data_interim          = proj + "data/"
fig                   = proj + "fig/"

raw                   = cf_space + "raw/"
processed             = cf_space + "processed/cf-Dunnsigouin_etal_2026/"

s2s_forecast_daily    = raw + 's2s/mars/ecmwf/forecast/sfc/daily/europe/' 
s2s_hindcast_daily    = raw + 's2s/mars/ecmwf/hindcast/sfc/daily/europe/'

era5_continuous_daily = raw + 'era5/continuous-format/europe/daily/'
era5_forecast_daily   = raw + 'era5/s2s-model-format/europe/forecast/daily/'
era5_hindcast_daily   = raw + 'era5/s2s-model-format/europe/hindcast/daily/'

dirs = {"proj":proj,
        "data_interim":data_interim,
        "fig":fig,
        "raw":raw,
        "processed":processed,
        "s2s_forecast_daily":s2s_forecast_daily,
        "s2s_hindcast_daily":s2s_hindcast_daily,
        "era5_continuous_daily":era5_continuous_daily,
        "era5_forecast_daily":era5_forecast_daily,
        "era5_hindcast_daily":era5_hindcast_daily,
}        
