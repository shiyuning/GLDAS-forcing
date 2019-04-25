#!/bin/sh

# Bash script to download GLDAS-2.1 data
# Author: Yuning Shi (yshi.at.psu.edu)

echo "##############################################"
echo "# GLDAS-forcing                              #"
echo "# https://github.com/shiyuning/GLDAS-forcing #"
echo "# Contact: Yuning Shi (yshi.at.psu.edu)      #"
echo "##############################################"

# Read configuration file
CONFIG_FILE=./forcing.config
chmod 700 $CONFIG_FILE
. $CONFIG_FILE

if [ $START_YEAR -lt 2010 ] ; then
    echo
    echo "Error: GLDAS-2.1 forcing data range from 01 Jan 2010 to present."
    echo "Please specify a valid START_YEAR in forcing.config and try again."
    echo
    exit
fi
if [ $START_MONTH -lt 1 -o $START_MONTH -gt 12 ] ; then
    echo
    echo "Error: Please specify a valid START_MONTH in forcing.config and try again."
    echo
    exit
fi
if [ $END_YEAR -lt $START_YEAR ] ; then
    echo
    echo "Error: END_YEAR is smaller than START_YEAR."
    echo "Please specify a valid END_YEAR in forcing.config and try again."
    echo
    exit
fi
if [ $END_MONTH -lt 1 -o $END_MONTH -gt 12 ] ; then
    echo
    echo "Error: Please specify a valid END_MONTH in forcing.config and try again."
    echo
    exit
fi
if [ $START_YEAR -eq $END_YEAR -a $END_MONTH -lt $START_MONTH ] ; then
    echo
    echo "Error: End date is earlier than start date."
    echo "Please specify a valid END_MONTH in forcing.config and try again."
    echo
    exit
fi

# Create a .netrc file in your home directory
touch ${HOME}/.netrc
echo "machine urs.earthdata.nasa.gov login $USER_NAME password $PASSWORD" > ${HOME}/.netrc
chmod 0600 ${HOME}/.netrc

# Create a cookie file
# This file will be used to persist sessions across calls to Wget or Curl
touch ${HOME}/.urs_cookies

# Run download script
echo
echo "Download starts."
. ./util/dl_gldas.sh download
echo "Download completed."
