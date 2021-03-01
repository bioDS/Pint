#!/bin/bash

git stash push --keep-index

ninja -C build test > test_output && cat test_output
ninja -C build coverage > coverage_output

ok=$(grep Ok test_output | grep [0-9] -o)
fail=$(grep "^Fail" test_output | grep [0-9] -o)
total_tests=$(echo $ok + $fail | bc -l)

total_coverage=$(grep "TOTAL" build/meson-logs/coverage.txt | grep -o "[0-9]*\%" | tr -d '%')

echo "coverage: $total_coverage%"

if (( $(echo "$total_coverage <= 50" | bc -l) )) ; then
	COLOR=red
elif (( $(echo "$total_coverage > 80" | bc -l) )); then
	COLOR=green
else
	COLOR=orange
fi

# coverage badge
badgename="coverage-$total_coverage-$COLOR.svg"

if [ ! -f badges/$badgename ]; then
	curl "https://img.shields.io/badge/coverage-$total_coverage%25-$COLOR" > badges/$badgename
fi

cp badges/$badgename coverage-badge.svg
git add coverage-badge.svg -f

# test results badge

if (( $(echo "$fail > $ok" | bc -l) )) ; then
	COLOR=red
elif (( $(echo "$fail > 0" | bc -l) )); then
	COLOR=orange
else
	COLOR=green
fi

badgename="results-$ok-$fail.svg"

if [ ! -f badges/$badgename ]; then
  curl "https://img.shields.io/badge/Tests-$ok%20out%20of%20$total_tests%20passed-$COLOR.svg" > badges/$badgename
fi

cp badges/$badgename test-badge.svg
git add test-badge.svg -f

git stash pop
