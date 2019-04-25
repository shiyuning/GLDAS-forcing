#!/bin/sh

# Function to convert from Julian day to date
Jul()
{
    date -d "$1-01-01 +$2 days -1 day" "+%Y-%m-%d";
}

operation=$1

# File extension to download/convert
if [ "$operation" == "download" ]; then
    ext=".nc4"
else
    echo "ERROR!"
fi

# Loop through the years to download data
start_date="$START_YEAR-$(printf "%2.2d" "$START_MONTH")-01"
end_date=$(date -d "$END_YEAR-$(printf "%2.2d" "$END_MONTH")-01 +1 month -1 day" "+%Y-%m-%d")
nod=$(( (`date -d $end_date +%s` - `date -d $start_date +%s`) / (24*3600) ))

for (( d=0; d<=$nod; d++ ))
do
    cyear=$(date -d "$start_date +$d days" "+%Y")
    cjday=$(date -d "$start_date +$d days" "+%j")

    nof=$(ls Data/$cyear/$cjday/GLDAS_NOAH025_3H.A$cyear*.021$ext 2>/dev/null | wc -l)
    if [ $cyear -eq 2000 -a $cjday -eq 001 ] ; then
        nof_avail=7
    else
        nof_avail=8
    fi

    if [ $nof -ne $nof_avail ] ; then
        if [ "$operation" == "download" ] ; then
            echo "Downloading $(Jul $cyear $cjday) data..."
            wget --load-cookies $HOME/.urs_cookies --save-cookies $HOME/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A $ext "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/$cyear/$cjday/" -P Data/$cyear/$cjday #&>/dev/null
        fi
    fi

done
