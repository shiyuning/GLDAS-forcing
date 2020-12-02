#!/usr/bin/env python3

import math
import sys
import os
import numpy as np
from datetime import timedelta, date, datetime
from netCDF4 import Dataset

def Closest(lat, lon, GLDAS_lat_masked, GLDAS_lon_masked, GLDAS_lat, GLDAS_lon):

    dist_masked = np.sqrt((GLDAS_lon_masked - lon)**2 +
                          (GLDAS_lat_masked - lat)**2)
    closest_masked = np.unravel_index(np.argmin(dist_masked, axis=None),
                                   dist_masked.shape)

    dist = np.sqrt((GLDAS_lon - lon)**2 + (GLDAS_lat - lat)**2)
    closest = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    if (abs(closest_masked[0] - closest[0]) > 1
        or abs(closest_masked[1] - closest[1]) > 1):
        land = 0
        print("%f, %f is not a land point" % (lat, lon))
    else:
        land = 1

    return closest[0], closest[1], land


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


def process_day(t, loc, path):

    '''
    Process one day of GLDAS data and convert it to Cycles input
    '''

    prcp  = [0.0] * len(loc)
    tx    = [-999.0] * len(loc)
    tn    = [999.0] * len(loc)
    wind  = [0.0] * len(loc)
    solar = [0.0] * len(loc)
    rhx   = [-999.0] * len(loc)
    rhn   = [999.0] * len(loc)
    data  = []
    counter = 0

    print(datetime.strftime(t, "%Y-%m-%d"))

    nc_path = '%s/%4.4d/%3.3d/' % (path,
                                   t.timetuple().tm_year,
                                   t.timetuple().tm_yday)

    for nc_name in os.listdir(nc_path):
        if nc_name.endswith(".nc4"):
            nc = Dataset(os.path.join(nc_path, nc_name), 'r')
            for i in range(len(loc)):
                (_prcp, _temp, _wind, _solar, _rh) = ReadVar(loc[i][0],
                                                             loc[i][1],
                                                             nc)

                prcp[i] += _prcp
                tx[i] = max(_temp, tx[i])
                tn[i] = min(_temp, tn[i])
                wind[i] += _wind
                solar[i] += _solar
                rhx[i] = max(_rh, rhx[i])
                rhn[i] = min(_rh, rhn[i])

            nc.close()

            counter += 1

    for i in range(len(loc)):
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

    # Read GLDAS grid data
    elevation_fp = data_path + '/GLDASp4_elevation_025d.nc4'
    nc = Dataset(elevation_fp, 'r')

    GLDAS_lat, GLDAS_lon = np.meshgrid(nc['lat'][:], nc['lon'][:],
                                       indexing='ij')
    GLDAS_lat_masked, GLDAS_lon_masked = np.meshgrid(nc['lat'][:], nc['lon'][:],
                                       indexing='ij')
    elev = nc['GLDAS_elevation'][0]
    elev = np.ma.filled(elev.astype(float), np.nan)


    GLDAS_lat_masked[np.isnan(elev)] = np.nan
    GLDAS_lon_masked[np.isnan(elev)] = np.nan

    filepath = 'location.txt'
    outfp = []
    loc = []
    fname = []

    with open(filepath) as fp:
        for _, line in enumerate(fp):
            li=line.strip()
            if not (li.startswith("#") or li.startswith("L") or (not li)):
                # Read lat/lon from location file
                strs = line.split()
                lat = float(strs[0])
                lon = float(strs[1])

                # Find the closest GLDAS grid
                _y, _x, land = Closest(lat, lon,
                                       GLDAS_lat_masked, GLDAS_lon_masked,
                                       GLDAS_lat, GLDAS_lon)

                if land == 0:
                    continue

                grid_lat = nc['lat'][_y]
                grid_lon = nc['lon'][_x]
                elevation = elev[_y][_x]

                lat_str = '%.2fS' %(abs(grid_lat)) if grid_lat < 0.0 \
                          else '%.2fN' %(abs(grid_lat))

                lon_str = '%.2fW' %(abs(grid_lon)) if grid_lon < 0.0 \
                          else '%.2fE' %(abs(grid_lon))

                # Check if grid is already in the list
                if [_y, _x] in loc:
                    if len(strs) == 3:
                        print('Site %s is in the same grid as %s.' %
                            (strs[2], fname[loc.index([_y, _x])]))
                    else:
                        print('Site %s, %s is in the same grid as %s.' %
                            (lat_str, lon_str, fname[loc.index([_y, _x])]))
                    continue

                # Add site to list
                loc.append([_y, _x])

                # Generate output file name
                if len(strs) == 3:
                    name = strs[2]
                    print('Processing data for %s, %s (%s)' % (lat_str,
                                                               lon_str,
                                                               name))
                    fname.append('gldas_' + name + '.weather')
                else:
                    print('Processing data for %s, %s' % (lat_str, lon_str))
                    fname.append('gldas' + lat_str + 'x' + lon_str + '.weather')

                # Open file and write header lines
                outfp.append(open(fname[-1], 'w'))
                outfp[-1].write('LATITUDE %.2f\n' % (grid_lat))
                outfp[-1].write('ALTITUDE %.2f\n' % (elevation))
                outfp[-1].write('SCREENING_HEIGHT 2\n')
                outfp[-1].write('YEAR    DOY     PP      TX      TN      '
                                'SOLAR   RHX     RHN     WIND\n')

    cday = start_date

    while cday <= end_date:
        data = process_day(cday, loc, data_path)

        [outfp[i].write(data[i]) for i in range(len(loc))]

        cday += timedelta(days=1)

    [outfp[i].close() for i in range(len(loc))]


main()

