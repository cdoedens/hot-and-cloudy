import sys
import os
import pvlib

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

sys.path.append('/home/548/pag548/code/aus-historical-solar_droughts/modules')

import logger 
from load_data import load_day

LOG = logger.get_logger(__name__)

def compute_timeseries_clearsky_ratio(date,
                                      TIMESERIES_DIR):
    """
    Compute a timeseries of ratio b/w tilting panel irradiance
    relative to clear sky conditions for a given day
    """

    # REZ mask
    mask_file = "/home/548/pag548/code/aus-historical-solar-droughts/data/boundary_files/REZ_mask.npz"
    loaded_mask = np.load(mask_file)
    mask = loaded_mask["mask"]

    date_dt = datetime.strptime(date, "%d-%m-%Y")

    LOG.info(f'Loading irradiance data for {date_dt}')
    dataset = load_day(resolution='p1s', date=date_dt)

    # apply REZ mask to lat/lon coordinates
    mask_da = xr.DataArray(mask, coords={"latitude": dataset.latitude, "longitude": dataset.longitude}, dims=["latitude", "longitude"])
    masked_ds = dataset.where(mask_da, drop=True)

    # get irradiance data, ensuring to flatten and remove all unnecessary nan values
    ghi = masked_ds.surface_global_irradiance.values.ravel()
    dni = masked_ds.direct_normal_irradiance.values.ravel()
    dhi = masked_ds.surface_diffuse_irradiance.values.ravel()
    nan_mask = np.isnan(ghi) # same for all vars
    ghi_clean = ghi[~nan_mask]
    dni_clean = dni[~nan_mask]
    dhi_clean = dhi[~nan_mask]

    # get correct time and coordinate data, so that it matches up with the remaining irradiance values
    lat_1d = masked_ds.latitude.values
    lon_1d = masked_ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")
    lat_grid_1d = lat_grid.ravel()
    lon_grid_1d = lon_grid.ravel()
    lat_1d_expanded = np.tile(lat_grid_1d, dataset.sizes["time"])  # Tile lat for all times
    lon_1d_expanded = np.tile(lon_grid_1d, dataset.sizes["time"])  # Tile lon for all times
    time_1d = np.repeat(masked_ds.time.values, len(lat_grid_1d))  # Repeat time for all lat/lon
    lat_1d_expanded_clean = lat_1d_expanded[~nan_mask]
    lon_1d_expanded_clean = lon_1d_expanded[~nan_mask]
    time_1d_clean = time_1d[~nan_mask]

    dataset.close()
        
    # calculate capacity factors using pvlib
    # the function defined in utils_V2 is essentially the same as the workflow in pv-output-tilting.ipynb
    LOG.info(f'Computing capacity factor for {date}')
    actual_ideal_ratio = tilting_panel_pr(
        pv_model = 'Canadian_Solar_CS5P_220M___2009_',
        inverter_model = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',
        ghi=ghi_clean,
        dni=dni_clean,
        dhi=dhi_clean,
        time=time_1d_clean,
        lat=lat_1d_expanded_clean,
        lon=lon_1d_expanded_clean
    )  

    # template to refit data to
    mask_template = masked_ds.surface_global_irradiance
    
    # Now need to get data back in line with coordinates
    # fill cf array with nan values so it can fit back into lat/lon coords
    filled = np.empty_like(ghi)
    # nan values outside the data
    filled[nan_mask] = np.nan
    # add the data to the same mask the input irradiance data was taken from
    filled[~nan_mask] = actual_ideal_ratio
    # convert data back into 3D xarray
    reshaped = filled.reshape(mask_template.shape)
    ratio_da = xr.DataArray(reshaped, coords=mask_template.coords, dims=mask_template.dims)

    mean_daily = ratio_da.mean(dim=["latitude", "longitude"], skipna=True)

    file_path = TIMESERIES_DIR
    os.makedirs(file_path, exist_ok=True)
    LOG.info(f'Writing data for {date} to {file_path}/{date}.nc')
    mean_daily.to_netcdf(f'{file_path}/{date}.nc')

    return


def tilting_panel_pr(
    pv_model,
    inverter_model,
    time,
    lat,
    lon,
    dni,
    ghi,
    dhi
    
):
    '''
    Other than pv and inverter models, all other arguments must be a flat 1D array of equal size
    '''
    
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
    # get panel/solar angles for a tilting panel system
    tracking = pvlib.tracking.singleaxis(
        apparent_zenith=solpos["apparent_zenith"],
        apparent_azimuth=solpos["azimuth"]
    )
    # compute airmass data
    airmass_relative = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'].values
    )
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(
        airmass_relative,
    )

    # copmute irradiances
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracking['surface_tilt'],
        surface_azimuth=tracking['surface_azimuth'],
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
        aoi=tracking['aoi'],
        module=module,
    )

    # compute ideal conditions
    linke_turbidity = np.maximum(2 + 0.1 * airmass_absolute, 2.5) # simplified parameterisation from ChatGPT as pvlib function was not working with array
    doy = pd.to_datetime(time).dayofyear
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)
    ideal_conditions = pvlib.clearsky.ineichen(
        apparent_zenith=solpos['apparent_zenith'].values,
        airmass_absolute=airmass_absolute,
        linke_turbidity=linke_turbidity,
        dni_extra=dni_extra
    )
    ideal_total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracking['surface_tilt'],
        surface_azimuth=tracking['surface_azimuth'],
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
        aoi=tracking['aoi'],
        module=module,
    )

    # Compute power outputs
    dc = pvlib.pvsystem.sapm(
        effective_irradiance=effective_irradiance.values,
        temp_cell=np.full_like(effective_irradiance, 18), # assume temperature of 18 deg C
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
        temp_cell=np.full_like(effective_irradiance, 18), # assume temperature of 18 deg C
        module=module
    )
    ac_ideal = pvlib.inverter.sandia(
        v_dc=dc_ideal['v_mp'],
        p_dc=dc_ideal['p_mp'],
        inverter=inverter
    )
    ac_ideal_QC = np.where(ac_ideal < 0, np.nan, ac_ideal)
    actual_ideal_ratio = ac_QC / ac_ideal_QC

    return actual_ideal_ratio

