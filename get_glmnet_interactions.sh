#!/usr/bin/bash
for i in `./glmnet.R | grep V`; do head -n1 test_int.csv | cut -d',' -f`echo $i | cut -d'V' -f2`; done
