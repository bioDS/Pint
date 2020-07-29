#!/bin/bash

trap "echo Exited!; exit;" SIGINT SIGTERM

rm fits_testing
rm simulated_data

ln -s ~/work/data/simulated_small_data simulated_data
ln -s fits_testing_small fits_testing

./fit_all.sh
