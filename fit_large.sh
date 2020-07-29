#!/bin/bash

trap "echo Exited!; exit;" SIGINT SIGTERM

rm fits_testing
rm simulated_data

ln -s ~/work/data/simulated_large_data simulated_data
ln -s fits_testing_large fits_testing

./fit_all.sh
