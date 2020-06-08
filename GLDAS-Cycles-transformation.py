#!/usr/bin/env python3

import math
import sys
import os
import numpy as np
from datetime import timedelta, date, datetime
from netCDF4 import Dataset

def Closest(lat, lon, path):

    elevation_fp = path + '/GLDASp4_elevation_025d.nc4'
    nc = Dataset(elevation_fp, 'r')

    best_y = (np.abs(nc.variables['lat'][:] - lat)).argmin()
    best_x = (np.abs(nc.variables['lon'][:] - lon)).argmin()

    return (best_y, best_x, nc['lat'][best_y], nc['lon'][best_x], nc['GLDAS_elevation'][0, best_y, best_x])


def ReadVar(y, x, nc_name):

    nc = Dataset(nc_name, 'r')

    _prcp = nc['Rainf_f_tavg'][0, y, x]
    _temp = nc['Tair_f_inst'][0, y, x]
    _wind = nc['Wind_f_inst'][0, y, x]
    _solar = nc['SWdown_f_tavg'][0, y, x]
    _pres = nc['Psurf_f_inst'][0, y, x]
    _spfh = nc['Qair_f_inst'][0, y, x]

    es = 611.2 * math.exp(17.67 * (_temp - 273.15) / (_temp - 273.15 + 243.5))
    ws = 0.622 * es / (_pres - es)
    w = _spfh / (1.0 - _spfh)
    _rh = w / ws
    if _rh > 1.0:
        _rh = 1.0

    return (_prcp, _temp, _wind, _solar, _rh)

def satvp(temp):
    return .6108 * math.exp(17.27 * temp / (temp + 237.3))


def ea(patm, q):
    return patm * q / (0.622 * (1 - q) + q)


def process_day(t, y, x, path):
    '''
    process one day of GLDAS data and convert it to Cycles input
    '''

    prcp = 0.0
    tx = -999.0
    tn = 999.0
    wind = 0.0
    solar = 0.0
    rhx = -999.0
    rhn = 999.0
    counter = 0

    print(datetime.strftime(t, "%Y-%m-%d"))

    nc_path = '%s/%4.4d/%3.3d/' %(path, t.timetuple().tm_year, t.timetuple().tm_yday)

    for nc_name in os.listdir(nc_path):
        if nc_name.endswith(".nc4"):
            nc = Dataset(os.path.join(nc_path, nc_name), 'r')
            (_prcp, _temp, _wind, _solar, _rh) = ReadVar(y, x, os.path.join(nc_path, nc_name))

            prcp += _prcp

            if _temp > tx:
                tx = _temp

            if _temp < tn:
                tn = _temp

            wind += _wind

            solar += _solar

            if _rh > rhx:
                rhx = _rh

            if _rh < rhn:
                rhn = _rh

            counter += 1

    prcp /= float(counter)
    prcp *= 86400.0

    wind /= float(counter)

    solar /= float(counter)
    solar *= 86400.0 / 1.0E6

    rhx *= 100.0
    rhn *= 100.0

    tx -= 273.15
    tn -= 273.15

    data = '%-16s%-8.4f%-8.2f%-8.2f%-8.4f%-8.2f%-8.2f%-8.2f\n' \
       %(t.strftime('%Y    %j'), prcp, tx, tn, solar, rhx, rhn, wind)

    return data


def main():

    if (len(sys.argv) != 4):
        print("Illegal number of parameters.")
        print("Usage: GLDAS-Cycles-transformation.py YYYY-MM-DD YYYY-MM-DD DATA_PATH")
        sys.exit(0)

    start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    data_path = sys.argv[3]

    filepath = 'location.txt'
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            li=line.strip()
            if not (li.startswith("#") or li.startswith("L")):
                    nums = line.split()
                    lat = float(nums[0])
                    lon = float(nums[1])

                    print('Processing data for {0}, {1}'.format(lat, lon))

                    (y, x, grid_lat, grid_lon, elevation) = Closest(lat, lon, data_path)

                    if grid_lat < 0.0:
                        lat_str = '%.2fS' %(abs(grid_lat))
                    else:
                        lat_str = '%.2fN' %(abs(grid_lat))

                    if grid_lon < 0.0:
                        lon_str = '%.2fW' %(abs(grid_lon))
                    else:
                        lon_str = '%.2fE' %(abs(grid_lon))

                    fname = 'met' + lat_str + 'x' + lon_str + '.weather'
                    outfp = open(fname, 'w')
                    outfp.write('LATITUDE %.2f\n' %(grid_lat))
                    outfp.write('ALTITUDE %.2f\n' %(elevation))
                    outfp.write('SCREENING_HEIGHT 2\n')
                    outfp.write('YEAR    DOY     PP      TX      TN     SOLAR      RHX      RHN     WIND\n')

                    cday = start_date

                    while cday <= end_date:
                        outfp.write(process_day(cday, y, x, data_path))
                        cday += timedelta(days = 1)

                    outfp.close()

    fp.close


main()

