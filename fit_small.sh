#!/bin/bash

trap "echo Exited!; exit;" SIGINT SIGTERM

for link in simulated_data fits_testing_adcalTRUE fits_testing_adcalFALSE; do
	if [[ -f $link ]]; then
		rm $link
	fi
done
mkdir -p fits_testing_small_adcalTRUE
mkdir -p fits_testing_small_adcalFALSE
ln -s ~/work/data/simulated_small_data simulated_data
ln -s fits_testing_small_adcalFALSE fits_testing_adcalFALSE
ln -s fits_testing_small_adcalTRUE fits_testing_adcalTRUE

./fit_all.sh 1 adcal
./fit_all.sh 1
