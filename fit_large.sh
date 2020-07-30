#!/bin/bash

trap "echo Exited!; exit;" SIGINT SIGTERM

for link in simulated_data fits_testing; do
	if [[ -L $link ]]; then
		rm $link
	fi
done

mkdir -p fits_testing
ln -s ~/work/data/simulated_large_data_sample simulated_data

./fit_all.sh 1 adcal
./fit_all.sh 1 limit_nbeta
./fit_all.sh 1
