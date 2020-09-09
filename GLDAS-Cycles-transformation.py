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

    best_lat, best_lon = nc['lat'][best_y], nc['lon'][best_y]
    elevation = nc['GLDAS_elevation'][0, best_y, best_x]

    nc.close()

    return (best_y, best_x, best_lat, best_lon, elevation)


def ReadVar(y, x, nc):

    _prcp  = nc['Rainf_f_tavg'][0, y, x]
    _temp  = nc['Tair_f_inst'][0, y, x]
    _wind  = nc['Wind_f_inst'][0, y, x]
    _solar = nc['SWdown_f_tavg'][0, y, x]
    _pres  = nc['Psurf_f_inst'][0, y, x]
    _spfh  = nc['Qair_f_inst'][0, y, x]

    es = 611.2 * math.exp(17.67 * (_temp - 273.15) / (_temp - 273.15 + 243.5))
    ws = 0.622 * es / (_pres - es)
    w = _spfh / (1.0 - _spfh)
    _rh = w / ws
    _rh = min(_rh, 1.0)

    return (_prcp, _temp, _wind, _solar, _rh)


def satvp(temp):

    return 0.6108 * math.exp(17.27 * temp / (temp + 237.3))


def ea(patm, q):

    return patm * q / (0.622 * (1.0 - q) + q)


def process_day(t, y, x, path):

    '''
    Process one day of GLDAS data and convert it to Cycles input
    '''

    prcp  = [0.0] * len(y)
    tx    = [-999.0] * len(y)
    tn    = [999.0] * len(y)
    wind  = [0.0] * len(y)
    solar = [0.0] * len(y)
    rhx   = [-999.0] * len(y)
    rhn   = [999.0] * len(y)
    data  = []
    counter = 0

    print(datetime.strftime(t, "%Y-%m-%d"))

    nc_path = '%s/%4.4d/%3.3d/' % (path,
                                   t.timetuple().tm_year,
                                   t.timetuple().tm_yday)

    for nc_name in os.listdir(nc_path):
        if nc_name.endswith(".nc4"):
            nc = Dataset(os.path.join(nc_path, nc_name), 'r')
            for i in range(len(y)):
                (_prcp, _temp, _wind, _solar, _rh) = ReadVar(y[i], x[i], nc)

                prcp[i] += _prcp
                tx[i] = max(_temp, tx[i])
                tn[i] = min(_temp, tn[i])
                wind[i] += _wind
                solar[i] += _solar
                rhx[i] = max(_rh, rhx[i])
                rhn[i] = min(_rh, rhn[i])

            nc.close()

            counter += 1

    for i in range(len(y)):
        prcp[i] /= float(counter)
        prcp[i] *= 86400.0

        wind[i] /= float(counter)

        solar[i] /= float(counter)
        solar[i] *= 86400.0 / 1.0E6

        rhx[i] *= 100.0
        rhn[i] *= 100.0

        tx[i] -= 273.15
        tn[i] -= 273.15

        data.append('%-16s%-8.4f%-8.2f%-8.2f%-8.3f%-8.2f%-8.2f%-8.2f\n' \
                    % (t.strftime('%Y    %j'),
                       prcp[i],
                       tx[i],
                       tn[i],
                       solar[i],
                       rhx[i],
                       rhn[i],
                       wind[i]))

    return data


def main():

    if (len(sys.argv) != 4):
        print("Illegal number of parameters.")
        print("Usage: GLDAS-Cycles-transformation.py "
              "YYYY-MM-DD YYYY-MM-DD DATA_PATH")
        sys.exit(0)

    start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    data_path = sys.argv[3]

    filepath = 'location.txt'
    y = []
    x = []
    outfp = []

    with open(filepath) as fp:
        for _, line in enumerate(fp):
            li=line.strip()
            if not (li.startswith("#") or li.startswith("L")):
                nums = line.split()
                lat = float(nums[0])
                lon = float(nums[1])

                print('Processing data for %s, %s' % (lat, lon))

                (_y, _x, grid_lat, grid_lon, elevation) = Closest(lat,
                                                                  lon,
                                                                  data_path)
                x.append(_x)
                y.append(_y)

                lat_str = '%.2fS' %(abs(grid_lat)) if grid_lat < 0.0 \
                          else '%.2fN' %(abs(grid_lat))

                lon_str = '%.2fW' %(abs(grid_lon)) if grid_lon < 0.0 \
                          else '%.2fE' %(abs(grid_lon))

                fname = 'gldas' + lat_str + 'x' + lon_str + '.weather'
                outfp.append(open(fname, 'w'))
                outfp[-1].write('LATITUDE %.2f\n' % (grid_lat))
                outfp[-1].write('ALTITUDE %.2f\n' % (elevation))
                outfp[-1].write('SCREENING_HEIGHT 2\n')
                outfp[-1].write('YEAR    DOY     PP      TX      TN      '
                                'SOLAR   RHX     RHN     WIND\n')

    cday = start_date

    while cday <= end_date:
        data = process_day(cday, y, x, data_path)

        [outfp[i].write(data[i]) for i in range(len(x))]

        cday += timedelta(days=1)

    [outfp[i].close() for i in range(len(x))]


main()

