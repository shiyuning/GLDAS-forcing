# GLDAS-forcing

GLDAS-forcing can **download** GLDAS-2.1 forcing data from NASA GES DISC archive.
In specific, the script downloads the GLDAS_NOAH025_3H, i.e., GLDAS Noah Land Surface Model L4 3 hourly 0.25 x 0.25 degree V2.1 data.
GLDAS-2.1 forcing is a combination of model and observation based forcing data sets (hereafter, GLDAS-2.1).
GLDAS-2.1 forcing is available from 1 January 2010 to now.
More capabilities, e.g., extracting GLDAS-2.1 forcing for PIHM and Cycles, may be added in the future.

## NASA Earthdata login for data access
NASA requires registration to access the GES DISC data.
Please follow [this link](https://wiki.earthdata.nasa.gov/display/EL/How+To+Register+With+Earthdata+Login) to register a new user in Earthdata login, and follow [this link](https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ) to authorize NASA GES DISC data archive in Earthdata login.

## Usage
1. Download the code from the [release page](https://github.com/shiyuning/GLDAS-forcing/releases).

2. Next, edit the configuration file `forcing.config`.
   Specify desired `START_YEAR`, `START_MONTH`, `END_YEAR`, and `END_MONTH`.
   Change `USERNAME` and `PASSWORD` to your Earthdata username and password.
3. To run the script, use

   ```shell
   $ ./gldas-forcing
   ```

**Note:** the script will detect whether `.nc4` files already exist so it only downloads when necessary.
Downloaded `.nc4` files will be stored in the `Data` directory, organized by year, and Julian day of year.
