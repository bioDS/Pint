#!/usr/bin/Rscript
library(ggplot2)
library(dplyr)

gdat = readRDS('fits_glinternet/dat_precrecf1_xyzFALSE.rds') %>%
                filter(test == "yes") %>%
                mutate(function_used="glinternet")
ldat_adcalfalse = readRDS('PrecRecF1/dat_precrecf1_lasso_adcalFALSE') %>% filter(test == "yes") %>%
                  mutate(function_used="test_lasso_adcalfalse")
ldat_adcaltrue = readRDS('PrecRecF1/dat_precrecf1_lasso_adcalTRUE') %>% #filter(test == "yes") %>%
                 mutate(function_used="test_lasso_adcaltrue")

vsglint_dat= bind_rows(ldat_adcalfalse, gdat)
adcal_dat = bind_rows(ldat_adcalfalse, ldat_adcaltrue)

vsglint_precision_plot = ggplot(vsglint_dat, aes(x=function_used, y=precision)) + geom_boxplot() + theme_bw()
vsglint_recall_plot = ggplot(vsglint_dat, aes(x=function_used, y=recall)) + geom_boxplot() + theme_bw()
vsglint_time_taken_plot = ggplot(vsglint_dat, aes(x=function_used, y=time_taken)) + scale_y_continuous(trans='log10') + geom_boxplot() + theme_bw()

ggsave(vsglint_precision_plot, file="plots/vsglint_precision_plot.pdf")
ggsave(vsglint_recall_plot, file="plots/vsglint_recall_plot.pdf")
ggsave(vsglint_time_taken_plot, file="plots/vsglint_time_plot.pdf")

adcal_precision_plot = ggplot(adcal_dat, aes(x=function_used, y=precision)) + geom_boxplot() + theme_bw()
adcal_recall_plot = ggplot(adcal_dat, aes(x=function_used, y=recall)) + geom_boxplot() + theme_bw()
adcal_time_taken_plot = ggplot(adcal_dat, aes(x=function_used, y=time_taken)) + scale_y_continuous(trans='log10') + geom_boxplot() + theme_bw()

ggsave(adcal_precision_plot, file="plots/adcal_precision_plot.pdf")
ggsave(adcal_recall_plot, file="plots/adcal_recall_plot.pdf")
ggsave(adcal_time_taken_plot, file="plots/adcal_time_plot.pdf")
