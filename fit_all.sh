#!/bin/bash

trap "echo Exited!; exit;" SIGINT SIGTERM

if [[ $1 ]]; then
	threads=$1
else
	threads=1
fi

adcal=''
fit_dir='fits_testing_adcalFALSE'
if [[ $2 == 'adcal' ]]; then
	adcal='adcal'
	fit_dir='fits_testing_adcalTRUE'
fi

counter=0

(
for f in `ls simulated_data`; do
	p=`expr "$f" : '.*p\([0-9]*\)_'`
	l=$(bc<<<"scale=1; x=sqrt($p) + 0.5; scale=0; x/1")
	if [[ -f $fit_dir/$f ]]; then
		echo "file '$f' already fitted, ignoring"
	else
		((i=i%threads)); ((i++==0)) && wait
		{ echo "$file '$f' not found, fitting now."
		taskset -c 0-15 ./fits_testing_only.R simulated_data/$f write $adcal || true; } &
	fi
done
)
