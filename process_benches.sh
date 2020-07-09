#!/bin/bash

for method in `ls compress_benches`; do
	wall_time=`cat compress_benches/$method/large_time_v_stats | grep "wall clock" | cut -d ":" -f5- | awk -F: '{ print  ($1 * 60) + $2 }';`
	lasso_time=`cat compress_benches/$method/large_time_v_term_output | grep "done in" | cut -d ' ' -f4`
	kbytes=`cat compress_benches/$method/large_time_v_stats | grep "Maximum resident" | cut -d' ' -f6`
	echo "$method, $wall_time, $lasso_time, $kbytes"
done
