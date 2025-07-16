import pvlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import os
import logger
import rasterio
from rasterstats import zonal_stats
import odc.geo.xr
from datetime import datetime
from pathlib import Path
from glob import glob
import dask.array as da

LOG = logger.get_logger(__name__)


# STRUCTURE OF FUNCTIONS
'''
SOLAR_WORKFLOW
    | READ_DATA
    |   | GET_REGION
    |   | _PREPROCESS
    |   | GET_FILES
    | CLEAR_SKY_PERFORMANCE
    |   | SOLAR_PV_GENERATION
    | SAVE_TIMESERIES
'''


def solar_workflow(date, region, tilt):
    ds = read_data(date, region)
    solar = clear_sky_performance(ds, tilt)
    save_timeseries(solar, date, region)
    del ds, solar
    LOG.info('END OF SOLAR_WORKFLOW')
    return

    
def read_data(date, region):
    LOG.info(f'reading data for {date}')
    # get correct file path based on the date
    if len(date) > 7:
        dir_dt = datetime.strptime(date, "%Y/%m/%d")
    else:
        dir_dt = datetime.strptime(date, "%Y/%m")
        
    if dir_dt <= datetime.strptime('2019-03-31', '%Y-%m-%d'):
        version = 'v1.0'
    else:
        version = 'v1.1'

    region_geo = get_region(region)
    lon_min, lat_min, lon_max, lat_max = region_geo.total_bounds

    # ensures lat_min etc. is available to _preprocess
    def _preprocess(ds):
        return ds[
        ['surface_global_irradiance', 'direct_normal_irradiance', 'surface_diffuse_irradiance']
        ].sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    chunk_size = 50
    LOG.info(f'chunking data with lat/lon size of: {chunk_size}')
    
    files = get_files(date, version)
    LOG.info(f"Loading {len(files)} files")
    ds = xr.open_mfdataset(
        files,
        preprocess=_preprocess,
        combine='by_coords',
        engine="h5netcdf",
    )
    ds = ds.chunk({'time':5, 'latitude':chunk_size, 'longitude':chunk_size})

    # apply mask
    LOG.info(f'applying regional mask')
    mask = rasterio.features.geometry_mask(
                region_geo.geometry,
                out_shape=ds.odc.geobox.shape,
                transform=ds.odc.geobox.affine,
                all_touched=False,
                invert=False)
    mask = xr.DataArray(~mask, dims=('latitude', 'longitude'),coords=dict(
            longitude=ds.longitude,
            latitude=ds.latitude)
                       ).chunk({'latitude':chunk_size, 'longitude':chunk_size})
   
    return ds.where(mask)

def get_files(date, version):
    directory = Path(f'/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/{version}/{date}')
    return sorted(str(p) for p in directory.rglob("*.nc"))
    
def get_region(region):
    # REGION SHOULD BE STRING WITH FORMAT "<TYPE>_<NAME>"
    region_type, region_name = region.split('_')
    LOG.info(f'getting region shape for: {region_type}, {region_name}')
    if region_type == 'REZ':
        
        shapefile = '/home/548/cd3022/aus-historical-solar-droughts/data/boundary_files/REZ-boundaries.shx'
        gdf = gpd.read_file(shapefile)
        
        zones_to_ignore = [ # no solar in these zones
            'Q1 Far North QLD',
            'N7 Tumut',
            'N8 Cooma-Monaro',
            'N10 Hunter Coast',
            'N11 Illawarra Coast',
            'V3 Western Victoria',
            'V4 South West Victoria',
            'V7 Gippsland Coast',
            'V8 Southern Ocean',
            'T4 North Tasmania Coast',
            'S1 South East SA',
            'S3 Mid-North SA',
            'S4 Yorke Peninsula',
            'S5 Northern SA',
            'S10 South East SA Coast'
        ]
        
        gdf = gdf[~gdf["Name"].isin(zones_to_ignore)]
    
        if region_name.upper() == 'ALL':
            return gdf
        else:
            return gdf[gdf["Name"].str.startswith(region_name)]
    
    if region_type == 'GCCSA':
        shapefile = '/home/548/cd3022/aus-historical-solar-droughts/data/boundary_files/GCCSA/GCCSA_2021_AUST_GDA2020.shp'
        gdf = gpd.read_file(shapefile)
        if region_name.upper() == 'ALL':
            return gdf
        else:
            return gdf[gdf['GCC_CODE21'] == region_name]
        
    else:
        LOG.info(f'unsuported region type "{region_type}" supplied')
        return

def clear_sky_performance(ds, tilt):

    LOG.info('reading dataset variables')
    ghi = ds.surface_global_irradiance.values.ravel()
    dni = ds.direct_normal_irradiance.values.ravel()
    dhi = ds.surface_diffuse_irradiance.values.ravel()
    nan_mask = np.isnan(ghi) # same for all vars
    ghi_clean = ghi[~nan_mask]
    dni_clean = dni[~nan_mask]
    dhi_clean = dhi[~nan_mask]

    # get correct time and coordinate data, so that it matches up with the remaining irradiance values
    lat_1d = ds.latitude.values
    lon_1d = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")
    lat_grid_1d = lat_grid.ravel()
    lon_grid_1d = lon_grid.ravel()
    lat_1d_expanded = np.tile(lat_grid_1d, ds.sizes["time"])  # Tile lat for all times
    lon_1d_expanded = np.tile(lon_grid_1d, ds.sizes["time"])  # Tile lon for all times
    time_1d = np.repeat(ds.time.values, len(lat_grid_1d))  # Repeat time for all lat/lon
    lat_1d_expanded_clean = lat_1d_expanded[~nan_mask]
    lon_1d_expanded_clean = lon_1d_expanded[~nan_mask]
    time_1d_clean = time_1d[~nan_mask]

    # BARRA-R2 temperature data
    LOG.info('getting BARRA-R2 temperature')
    year = pd.to_datetime(ds.isel(time=50).time.values.item()).year
    month = pd.to_datetime(ds.isel(time=50).time.values.item()).month

    barra_file = f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/tas/latest/tas_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'
    barra = xr.open_dataset(
        barra_file,
        engine='h5netcdf'
    )
    LOG.info('BARRA file opened')
    points = xr.Dataset(
        {
            "latitude": ("points", lat_1d_expanded_clean),
            "longitude": ("points", lon_1d_expanded_clean),
            "time": ("points", time_1d_clean),
        }
    )
    temp_barra = barra["tas"].sel(
        lat=points["latitude"],
        lon=points["longitude"],
        time=points["time"],
        method="nearest"
    )
    temp_clean = temp_barra.values - 273.15

    # calculate capacity factors using pvlib
    LOG.info(f'running pvlib functions')
    actual, ideal = solar_pv_generation(
        pv_model = 'Canadian_Solar_CS5P_220M___2009_',
        inverter_model = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',
        ghi=ghi_clean,
        dni=dni_clean,
        dhi=dhi_clean,
        time=time_1d_clean,
        lat=lat_1d_expanded_clean,
        lon=lon_1d_expanded_clean,
        temp=temp_clean,
        tilt=tilt
    )

    mask_template = ds.surface_global_irradiance
    actual_filled = np.empty_like(ghi)
    ideal_filled = np.empty_like(ghi)

    actual_filled[nan_mask] = np.nan
    actual_filled[~nan_mask] = actual

    ideal_filled[nan_mask] = np.nan
    ideal_filled[~nan_mask] = ideal

    actual_reshaped = actual_filled.reshape(mask_template.shape)
    ideal_reshaped = ideal_filled.reshape(mask_template.shape)
    
    return xr.Dataset(
        data_vars={
            "actual": (mask_template.dims, actual_reshaped),
            "ideal": (mask_template.dims, ideal_reshaped)
        },
        coords=mask_template.coords,
        )


    
def save_timeseries(ds, date, region):

    region_type, region_name = region.split('_')

    nem_timeseries = ds.mean(dim=["latitude", "longitude"], skipna=True)
    year=date[:4]
    month=date[5:7]
    day=date[8:10]

    file_name = f'{region_name}_timeseries_{year}-{month}-{day}.nc'
    file_path = f'/g/data/gb02/cd3022/hot-and-cloudy/solar-pv/{region_type}/{region_name}/{year}/{month}/'
    
    os.makedirs(file_path, exist_ok=True)
    LOG.info(f'Writing data to: {file_path}/{file_name}')
    nem_timeseries.to_netcdf(f'{file_path}/{file_name}')
    return


def solar_pv_generation(
    pv_model,
    inverter_model,
    time,
    lat,
    lon,
    dni,
    ghi,
    dhi,
    temp,
    tilt
):
    '''
    Other than pv and inverter models, all other arguments must be a flat 1D array of equal size
    '''

    if tilt not in ['fixed', 'single_axis']:
        raise ValueError(f'Unrecognised tilt: {tilt}. tilt must be "fixed" or "single_axis"')
    
    # get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules[pv_model]
    inverter = sapm_inverters[inverter_model]

    # Compute solar position for all grid cells at once
    solpos = pvlib.solarposition.get_solarposition(
        time,
        lat,
        lon,
    )

    # TILTING VS FIXED AXIS VALUES
    if tilt == 'single_axis':  
        # get panel/solar angles for a tilting panel system
        tracking = pvlib.tracking.singleaxis(
            apparent_zenith=solpos["apparent_zenith"],
            apparent_azimuth=solpos["azimuth"]
        )
        surface_tilt = tracking['surface_tilt']
        surface_azimuth = tracking['surface_azimuth']
        aoi = tracking['aoi']
    
    elif tilt == 'fixed':
        # Find the angle of incidence
        surface_tilt = -lat.ravel()
        surface_azimuth = [0] * len(lat)

        aoi = pvlib.irradiance.aoi(
            surface_tilt=surface_tilt.data,
            surface_azimuth=surface_azimuth,
            solar_zenith=solpos["apparent_zenith"],
            solar_azimuth=solpos["azimuth"],
        )

    # compute airmass data
    airmass_relative = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'].values
    )
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(
        airmass_relative,
    )

    # compute irradiances
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos['apparent_zenith'].values,
        solar_azimuth=solpos['azimuth'].values
    )
    
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        poa_direct=total_irradiance['poa_direct'],
        poa_diffuse=total_irradiance['poa_diffuse'],
        airmass_absolute=airmass_absolute,
        aoi=aoi,
        module=module,
    )

    # compute ideal conditions
    linke_turbidity = np.maximum(2 + 0.1 * airmass_absolute, 2.5)
    doy = pd.to_datetime(time).dayofyear
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)
    ideal_conditions = pvlib.clearsky.ineichen(
        apparent_zenith=solpos['apparent_zenith'].values,
        airmass_absolute=airmass_absolute,
        linke_turbidity=linke_turbidity,
        dni_extra=dni_extra
    )
    ideal_total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=ideal_conditions['dni'],
        ghi=ideal_conditions['ghi'],
        dhi=ideal_conditions['dhi'],
        solar_zenith=solpos['apparent_zenith'].values,
        solar_azimuth=solpos['azimuth'].values
    )
        
    ideal_effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        poa_direct=ideal_total_irradiance['poa_direct'],
        poa_diffuse=ideal_total_irradiance['poa_diffuse'],
        airmass_absolute=airmass_absolute,
        aoi=aoi,
        module=module,
    )

    # Compute power outputs
    dc = pvlib.pvsystem.sapm(
        effective_irradiance=effective_irradiance.values,
        temp_cell=temp,
        module=module
    )
    
    ac = pvlib.inverter.sandia(
        v_dc=dc['v_mp'],
        p_dc=dc['p_mp'],
        inverter=inverter
    )
    ac_QC = np.where(ac < 0, np.nan, ac)
    
    # ideal power output
    dc_ideal = pvlib.pvsystem.sapm(
        effective_irradiance=ideal_effective_irradiance.values,
        temp_cell=temp, # assume temperature of 18 deg C
        module=module
    )
    ac_ideal = pvlib.inverter.sandia(
        v_dc=dc_ideal['v_mp'],
        p_dc=dc_ideal['p_mp'],
        inverter=inverter
    )
    ac_ideal_QC = np.where(ac_ideal < 0, np.nan, ac_ideal)

    return ac_QC, ac_ideal_QC
        