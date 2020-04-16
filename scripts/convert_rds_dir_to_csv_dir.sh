#!/usr/bin/bash


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

echo "converting from '$1' to '$2'"

for file in `ls $1`; do
    /home/kieran/work/lasso_testing/scripts/rds_to_csv.R "$1/$file" $2
done