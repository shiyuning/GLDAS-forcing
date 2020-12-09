#!/usr/bin/env python3

import math
import sys
import os
import numpy as np
from datetime import timedelta, date, datetime
from netCDF4 import Dataset

def closest_grid(lat, lon, lat_array, lon_array):
    '''Find closest grid to an input site
    '''
    dist = np.sqrt((lon_array - lon)**2 + (lat_array - lat)**2)
    closest = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    return closest


def find_grid(site, lat, lon,
              GLDAS_lat_masked, GLDAS_lon_masked,
              GLDAS_lat, GLDAS_lon):
    '''Find closet land grid to an input site
    This function finds the closet unmasked grid and the closet masked grid to
    the specified site. By comparing the two grids, it will determine if the
    specified is a land point.
    '''
    closest_masked = closest_grid(lat, lon, GLDAS_lat_masked, GLDAS_lon_masked)
    closest_unmasked = closest_grid(lat, lon, GLDAS_lat, GLDAS_lon)

    if (abs(closest_masked[0] - closest_unmasked[0]) > 1 or
        abs(closest_masked[1] - closest_unmasked[1]) > 1):
        land = 0
        print("Cannot find nearest land grid to %s." % (site))
    else:
        land = 1
        if (closest_masked[0] != closest_unmasked[0] or
            closest_masked[1] != closest_unmasked[1]):
            print('Nearest GLDAS grid to %s is not a land point. '
                  'A nearest land point is chosen instead.' % (site))

    ind = np.ravel_multi_index([closest_masked[0], closest_masked[1]],
                               GLDAS_lat.shape)

    return ind, land


def read_var(grids, nc):
    '''Read meteorological variables of an array of desired grids from netCDF
    The netCDF variable arrays are flattened to make reading faster
    '''
    prcp  = nc['Rainf_f_tavg'][0].flatten()[grids]
    tmp  = nc['Tair_f_inst'][0].flatten()[grids]
    wind  = nc['Wind_f_inst'][0].flatten()[grids]
    solar = nc['SWdown_f_tavg'][0].flatten()[grids]
    pres  = nc['Psurf_f_inst'][0].flatten()[grids]
    spfh  = nc['Qair_f_inst'][0].flatten()[grids]

    es = 611.2 * np.exp(17.67 * (tmp - 273.15) / (tmp - 273.15 + 243.5))
    ws = 0.622 * es / (pres - es)
    w = spfh / (1.0 - spfh)
    rh = w / ws
    rh = np.minimum(rh, np.ones(rh.shape))

    return prcp, tmp, wind, solar, rh


def satvp(temp):
    '''Calculate saturation vapor pressure
    '''
    return 0.6108 * math.exp(17.27 * temp / (temp + 237.3))


def process_day(t, grids, path, fp):
    '''Process one day of GLDAS data and convert it to Cycles input
    '''

    daily_prcp = [0.0] * len(grids)
    tx = [-999.0] * len(grids)
    tn = [999.0] * len(grids)
    daily_wind = [0.0] * len(grids)
    daily_solar = [0.0] * len(grids)
    rhx = [-999.0] * len(grids)
    rhn = [999.0] * len(grids)
    data = []
    counter = 0

    print(datetime.strftime(t, "%Y-%m-%d"))

    nc_path = '%s/%4.4d/%3.3d/' % (path,
                                   t.timetuple().tm_year,
                                   t.timetuple().tm_yday)

    for nc_name in os.listdir(nc_path):
        if nc_name.endswith(".nc4"):
            with Dataset(os.path.join(nc_path, nc_name), 'r') as nc:
                prcp, tmp, wind, solar, rh = read_var(grids, nc)

                daily_prcp += prcp
                tx = np.maximum(tmp, tx)
                tn = np.minimum(tmp, tn)
                daily_wind += wind
                daily_solar += solar
                rhx = np.maximum(rh, rhx)
                rhn = np.minimum(rh, rhn)

            counter += 1

    daily_prcp *= 86400.0 / float(counter)
    daily_wind /= float(counter)
    daily_solar *= 86400.0 / 1.0E6 / float(counter)

    rhx *= 100.0
    rhn *= 100.0

    tx -= 273.15
    tn -= 273.15

    for i in range(len(grids)):
        fp[i].write('%-16s%-8.4f%-8.2f%-8.2f%-8.3f%-8.2f%-8.2f%-8.2f\n' \
                     % (t.strftime('%Y    %j'),
                        daily_prcp[i],
                        tx[i],
                        tn[i],
                        daily_solar[i],
                        rhx[i],
                        rhn[i],
                        daily_wind[i]))


def main():
    '''Generate Cycles weather files from GLDAS forcing for the locations
    specified in a location.txt file
    '''
    # Read start/end dates and path to GLDAS files from command line
    if (len(sys.argv) != 4):
        sys.exit("Illegal number of parameters."
                 "Usage: GLDAS-Cycles-transformation.py "
                 "YYYY-MM-DD YYYY-MM-DD DATA_PATH")

    start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    data_path = sys.argv[3]

    # Read GLDAS grid and elevation data
    with Dataset(data_path + '/GLDASp4_elevation_025d.nc4', 'r') as nc:
        elev_array = nc['GLDAS_elevation'][0]
        elev_array = np.ma.filled(elev_array.astype(float), np.nan)

        GLDAS_lat, GLDAS_lon = np.meshgrid(nc['lat'][:], nc['lon'][:],
                                           indexing='ij')
        GLDAS_lat_masked, GLDAS_lon_masked = np.meshgrid(nc['lat'][:],
                                                         nc['lon'][:],
                                                         indexing='ij')

    # Mask sea grids lat/lon as nan
    GLDAS_lat_masked[np.isnan(elev_array)] = np.nan
    GLDAS_lon_masked[np.isnan(elev_array)] = np.nan

    sites = []
    grids = []

    # Create weather directory for created weather files
    if not os.path.exists('weather'):
        os.makedirs('weather')

    # Read locations from file
    with open('location.txt') as fp:
        for line in fp:
            line = line.strip()

            # Skipe header line, comment lines and empty lines
            if (line.startswith("#") or line.startswith("L") or (not line)):
                continue

            # Read lat/lon from location file
            strs = line.split()
            lat = float(strs[0])
            lon = float(strs[1])

            if len(strs) == 3:          # Site name is defined
                site_name = strs[2]
            else:                       # Site name is not defined
                site_name = '%.3f%sx%.3f%s' % (abs(lat),
                                               'S' if lat < 0.0 else 'N',
                                               abs(lon),
                                               'W' if lon < 0.0 else 'E')

            # Find the closest GLDAS grid
            grid_ind, land = find_grid(site_name, lat, lon,
                                       GLDAS_lat_masked, GLDAS_lon_masked,
                                       GLDAS_lat, GLDAS_lon)

            # Skip sea grids
            if not land:
                continue

            # Check if grid is already in the list
            if grid_ind in grids:
                print('Site %s is in the same grid as %s.' %
                      (site_name, sites[grids.index(grid_ind)]))
                continue

            # Add site to list
            sites.append(site_name)
            grids.append(grid_ind)

    weather_fp = []

    for i, g in enumerate(grids):
        # Get lat/lon and elevation of nearest grid
        grid_lat = GLDAS_lat.flatten()[g]
        grid_lon = GLDAS_lon.flatten()[g]
        elevation = elev_array.flatten()[g]

        # Open weather file and write header lines
        fn = 'weather/GLDAS_' + sites[i] + '.weather'
        weather_fp.append(open(fn, 'w', buffering=1))
        weather_fp[-1].write('# GLDAS grid %.3f%sx%.3f%s\n'
                             % (abs(grid_lat), 'S' if grid_lat < 0.0 else 'N',
                                abs(grid_lon), 'W' if grid_lon < 0.0 else 'E'))
        weather_fp[-1].write('%-20s%.2f\n' % ('LATITUDE', grid_lat))
        weather_fp[-1].write('%-20s%.2f\n' % ('ALTITUDE', elevation))
        weather_fp[-1].write('%-20s%.1f\n' % ('SCREENING_HEIGHT', 2.0))
        weather_fp[-1].write('%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-s\n' %
                             ('YEAR', 'DOY', 'PP', 'TX', 'TN', 'SOLAR',
                              'RHX', 'RHN', 'WIND'))

    # Read GLDAS data and convert to Cycles weathare files
    cday = start_date
    while cday <= end_date:
        # Process each day's data
        process_day(cday, grids, data_path, weather_fp)

        cday += timedelta(days=1)

    # Close weather fiels
    [f.close() for f in weather_fp]


if __name__ == '__main__':
    main()

