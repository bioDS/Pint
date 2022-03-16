#!/bin/bash

if [ ! $(git diff --staged | wc -l) -gt 0 ]; then
	exit
fi
isdiff=0
if [ $(git diff | wc -l) -gt 0 ]; then
	isdiff=1
fi

if [ $isdiff -eq 1 ]; then
	git commit --quiet --no-verify -m "temp for stash-working" && \
	git stash push "$@" && \
	git reset --quiet --soft HEAD~1
fi

test_output=$(ninja -C build test -j 1)
coverage_output=$(ninja -C build coverage)

ok=$(echo $test_output | grep  "Ok:\s\s*[0-9][0-9]*" -o | grep "[0-9][0-9]*" -o)
fail=$(echo $test_output | grep -P "(?<!Expected\s)Fail:\s\s*[0-9][0-9]*" -o | grep "[0-9][0-9]*" -o)

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

echo "$ok passed"
echo "$fail failed"

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

if [ $isdiff -eq 1 ]; then
	git stash pop
fi

echo $test_output > test_output
echo $coverage_output > coverage_output
