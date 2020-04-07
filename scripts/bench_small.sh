#!/usr/bin/bash
set -x #echo on

#BENCHMARK_SETS=/home/kieran/work/data/simulated_data_small_repeat
BENCHMARK_SETS=/home/kieran/work/lasso_testing/bench_sets_small
BASE_DIR=/home/kieran/work/lasso_testing
OUTPUT_DIR=/home/kieran/work/lasso_testing/bench_runs

cd $BASE_DIR
#current_run_dir="bench_runs/$(date)"
mkdir -p "$current_run_dir"

for bench_set in `ls $BENCHMARK_SETS`; do
    echo $bench_set
    x="$BENCHMARK_SETS/$bench_set/X.csv"
    y="$BENCHMARK_SETS/$bench_set/Y.csv"

    echo "X: $x"
    echo "Y: $y"

    rows=`wc -l $y | cut -d' ' -f1`
    cols=`head -n1 $x | cut -d'"' -f3- | tr ',' ' ' | wc -w`

    echo "cols: $cols"
    echo "rows: $rows"

    ./mergefind-release/src/lasso_exe $x $y cyclic int F 1 $rows $cols > "$OUTPUT_DIR/$bench_set.log"
done
