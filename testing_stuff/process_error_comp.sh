for correction in "c" "nc"; do
	for threads in "1" "8"; do
		echo "processing ${threads}_${correction}"
		rg iters error_comp_cost.txt | rg "^$correction, $threads:" | cut -d ':' -f3  > ${threads}_${correction}.csv
		rg "total time" error_comp_cost.txt | rg "^$correction, $threads:" | cut -d' ' -f5  | sed 's/,/, /g' > ${threads}_${correction}_total_times.csv
	done
done
rg "^using multiplier" error_comp_cost.txt | sed '1~2d' | sed '1~2d' | cut -d':' -f2  > mults.csv
rg "blocks of size" error_comp_cost.txt | rg "nc" | rg "8:" | cut -d ' ' -f9  > 8_blocksizes.csv
rg "blocks of size" error_comp_cost.txt | rg "nc" | rg "1:" | cut -d ' ' -f9  > 1_blocksizes.csv
rg "^c, 8:.*correction time" error_comp_cost.txt | cut -d' ' -f6- | sed "s/ (/, /" | sed "s/)//" > 8_correction_times.csv
rg "^c, 1:.*correction time" error_comp_cost.txt | cut -d' ' -f6- | sed "s/ (/, /" | sed "s/)//" > 1_correction_times.csv
