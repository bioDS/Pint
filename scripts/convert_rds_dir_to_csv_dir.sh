#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -z $1 ]; then
    echo "usage convert_rds_dir_to_csv_dir.sh [from] [to]"
    echo "must supply from directory"
    exit
fi
if [ ! -d $1 ]; then
    echo "from directory must exist"
    exit
fi;
if [ -z $2 ]; then
    echo "usage convert_rds_dir_to_csv_dir.sh [from] [to]"
    echo "must supply to directory"
    exit
fi
if [ ! -d $2 ]; then
    echo "to directory must exist"
    exit
fi;

if [ ! -z $3 ]; then
    limit=$3
else
    ((limit=-1))
fi;

echo "converting from '$1' to '$2'"

for file in `ls $1`; do
    if [ $limit -gt 0 ]; then
        echo "$limit iterations remaining"
    fi
    $SCRIPT_DIR/rds_to_csv.R "$1/$file" $2
    ((limit=limit-1))
    if [ $limit -eq 0 ]; then
        echo "halting after reaching iteration limit"
        break;
    fi;
done
