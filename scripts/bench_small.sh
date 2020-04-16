#!/usr/bin/bash
#set -x #echo on

#BENCHMARK_SETS=/home/kieran/work/data/simulated_data_small_repeat
BENCHMARK_SETS=/home/kieran/work/lasso_testing/bench_sets_small
BASE_DIR=/home/kieran/work/lasso_testing
OUTPUT_DIR=/home/kieran/work/lasso_testing/bench_runs
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LAMBDA_VALUE=0.1

bash $SCRIPT_DIR/bench.sh $BENCHMARK_SETS $BASE_DIR $OUTPUT_DIR $1