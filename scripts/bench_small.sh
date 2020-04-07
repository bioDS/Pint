#!/usr/bin/bash
#set -x #echo on

#BENCHMARK_SETS=/home/kieran/work/data/simulated_data_small_repeat
BENCHMARK_SETS=/home/kieran/work/lasso_testing/bench_sets_small
BASE_DIR=/home/kieran/work/lasso_testing
OUTPUT_DIR=/home/kieran/work/lasso_testing/bench_runs

LAMBDA_VALUE=0.1

if [ -z $1 ]; then
    echo "must supply a test name as an argument"
    exit
fi

test_name=$1

cd $BASE_DIR
mkdir -p $OUTPUT_DIR/$test_name

for bench_set in `ls $BENCHMARK_SETS`; do
    echo "running $bench_set"
    x="$BENCHMARK_SETS/$bench_set/X.csv"
    y="$BENCHMARK_SETS/$bench_set/Y.csv"

    rows=`wc -l $y | cut -d' ' -f1`
    cols=`head -n1 $x | cut -d'"' -f3- | tr ',' ' ' | wc -w`

    ./mergefind-release/src/lasso_exe $x $y cyclic int F $LAMBDA_VALUE $rows $cols > "$OUTPUT_DIR/$1/$bench_set.log"
done
