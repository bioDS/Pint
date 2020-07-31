#!/bin/bash

trap "echo Exited!; exit;" SIGINT SIGTERM

if [[ $1 ]]; then
	threads=$1
else
	threads=1
fi

adcal='no'
adcalstr='FALSE'
fit_dir='fits_testing'
if [[ $2 == 'adcal' ]]; then
	adcal='adcal'
	adcalstr='TRUE'
fi
limit_nbeta='no_limit'
nbetastr='-1'
if [[ $3 == 'limit_nbeta' ]]; then
	limit_nbeta='limit_nbeta'
	nbetastr='2000'
fi


counter=0

(
for f in `ls simulated_data`; do
	fit_f=`expr "$f" : '\(.*_\)'`"adcal${adcalstr}_nbetalimit${nbetastr}"`expr "$f" : '.*\(_.*\)'`
	p=`expr "$f" : '.*p\([0-9]*\)_'`
	l=$(bc<<<"scale=1; x=sqrt($p) + 0.5; scale=0; x/1")
	if [[ -f $fit_dir/$fit_f ]]; then
		echo "file '$fit_f' already fitted, ignoring"
	else
		((i=i%threads)); ((i++==0)) && wait
		{ echo "file '$fit_f' not found, fitting now."
		taskset -c 0-15 ./fits_testing_only.R simulated_data/$f write $adcal $limit_nbeta || true; } &
	fi
done
)
