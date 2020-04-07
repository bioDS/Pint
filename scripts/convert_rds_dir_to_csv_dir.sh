#!/usr/bin/bash

echo "converting from '$1' to '$2'"

for file in `ls $1`; do
    /home/kieran/work/lasso_testing/scripts/rds_to_csv.R "$1/$file" $2
done