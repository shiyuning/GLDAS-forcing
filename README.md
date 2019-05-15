# GLDAS-forcing

GLDAS-forcing can **download** GLDAS-2.1 forcing data from NASA GES DISC archive, and **generate** Cycles weather files for given locations using GLDAS-2.1 forcing data.
In specific, the script downloads the GLDAS_NOAH025_3H, i.e., GLDAS Noah Land Surface Model L4 3 hourly 0.25 x 0.25 degree V2.1 data.
GLDAS-2.1 forcing is a combination of model and observation based forcing data sets (hereafter, GLDAS-2.1).
GLDAS-2.1 forcing is available from 1 January 2000 to now.

## Usage
1. Download the code from the [release page](https://github.com/shiyuning/GLDAS-forcing/releases).

2. To download GLDAS-2.1 forcing, run
   ```shell
   $ ./dl_gldas_forcing.sh YYYY-MM YYYY-MM DOWNLOAD_PATH
   ```
   The first `YYYY-MM` indicates the start year and month, and the second `YYYY-MM` indicates the end year and month. Data will be downloaded to the specified `DOWNLOAD_PATH`.

   **Note:** the script will detect whether `.nc4` files already exist so it only downloads when necessary.
3. To generate Cycles weather file from GLDAS-2.1 forcing, run
   ```shell
   $ python ./GLDAS-Cycles-transformation.py YYYY-MM-DD YYYY-MM-DD PATH_TO_NETCDF4_DATA
   ```
   The first `YYYY-MM-DD` indicates the start year, month, and date, and the second `YYYY-MM-DD` indicates the end year, month, and date. `PATH_TO_NETCDF4_DATA` is the directory where netCDF4 data are stored, which should be the `DOWNLOAD_PATH` in Step 2. The desired locations should be added to the `location.txt` file.

   **NOTE:** the `GLDAS-Cycles-transpormation.py` script requires [Python netCDF4 module](https://unidata.github.io/netcdf4-python/netCDF4/index.html), which can be installed in the [Anaconda environment](https://anaconda.org/anaconda/netcdf4).
