#!/bin/sh

# Bash script to download GLDAS-2.1 data
# Author: Yuning Shi (yshi.at.psu.edu)

Jul()
{
    date -d "$1-01-01 +$2 days -1 day" "+%Y-%m-%d";
}

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters."
    echo "Usage: \$./dl_gldas_forcing YYYY-MM YYYY-MM DATA_PATH"
    exit
fi

START_YEAR=$(echo $1 | awk -F"-" '{ print $1}')
START_MONTH=$(echo $1 | awk -F"-" '{ print $2}')
END_YEAR=$(echo $2 | awk -F"-" '{ print $1}')
END_MONTH=$(echo $2 | awk -F"-" '{ print $2}')
DATA_PATH=$3

if [ $START_YEAR -lt 2000 ] ; then
    echo
    echo "Error: GLDAS-2.1 forcing data range from 01 Jan 2000 to present."
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

# Create a cookie file
# This file will be used to persist sessions across calls to Wget or Curl
touch ${HOME}/.urs_cookies

# Run download script
echo
echo "Download starts."

echo "Download GLDAS elevation data..."
wget https://ldas.gsfc.nasa.gov/sites/default/files/ldas/gldas/ELEV/GLDASp4_elevation_025d.nc4 -P $DATA_PATH &>/dev/null

# Loop through the years to download data
start_date="$START_YEAR-$(printf "%2.2d" "$START_MONTH")-01"
end_date=$(date -d "$END_YEAR-$(printf "%2.2d" "$END_MONTH")-01 +1 month -1 day" "+%Y-%m-%d")
nod=$(( (`date -d $end_date +%s` - `date -d $start_date +%s`) / (24*3600) ))

for (( d=0; d<=$nod; d++ ))
do
    cyear=$(date -d "$start_date +$d days" "+%Y")
    cjday=$(date -d "$start_date +$d days" "+%j")

    nof=$(ls $DATA_PATH/$cyear/$cjday/GLDAS_NOAH025_3H.A$cyear*.021.ch4 2>/dev/null | wc -l)
    if [ $cyear -eq 2000 -a $cjday -eq 001 ] ; then
        nof_avail=7
    else
        nof_avail=8
    fi

    if [ $nof -ne $nof_avail ] ; then
        echo "Downloading $(Jul $cyear $cjday) data..."
        wget --load-cookies $HOME/.urs_cookies --save-cookies $HOME/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A ".nc4" "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/$cyear/$cjday/" -P $DATA_PATH/$cyear/$cjday &>/dev/null
    fi
done

echo "Download completed."
