#!/usr/bin/Rscript
library(ggplot2)
library(dplyr)

gdat = readRDS('fits_glinternet/dat_precrecf1_xyzFALSE.rds') %>%
                filter(test == "yes") %>%
                mutate(function_used="glinternet")
ldat_small_adcalfalse = readRDS('PrecRecF1/dat_precrecf1_lasso') %>%
						filter(n==10000) %>% filter(adcal==FALSE) %>% filter(test == "yes") %>%
                  mutate(function_used="Exhaustive Lasso")
ldat_small_adcaltrue  = readRDS('PrecRecF1/dat_precrecf1_lasso') %>%
						filter(n==10000) %>% filter(adcal==TRUE) %>% filter(test=="yes") %>%
                 mutate(function_used="Lasso w/ Adcal")
ldat_large_adcaltrue  = readRDS('PrecRecF1/dat_precrecf1_lasso') %>%
						filter(n==10000) %>% filter(adcal==TRUE) %>% filter(test=="yes") %>%
                 mutate(function_used="Lasso w/ Adcal")
ldat_large_limit_nb   = readRDS('PrecRecF1/dat_precrecf1_lasso') %>%
						filter(n==10000) %>% filter(adcal==FALSE) %>% filter(nbeta_limit == 2000) %>% filter(test=="yes") %>%
                 mutate(function_used="Limited beta lasso")

vsglint_dat= bind_rows(ldat_small_adcalfalse, gdat)
adcal_dat = bind_rows(ldat_small_adcalfalse, ldat_small_adcaltrue)
large_adcal_limit_nb_dat = bind_rows(ldat_large_adcaltrue, ldat_large_limit_nb)

vsglint_precision_plot = ggplot(vsglint_dat, aes(x=function_used, y=precision)) + geom_boxplot() + theme_bw()
vsglint_recall_plot = ggplot(vsglint_dat, aes(x=function_used, y=recall)) + geom_boxplot() + theme_bw()
vsglint_time_taken_plot = ggplot(vsglint_dat, aes(x=function_used, y=time_taken)) + scale_y_continuous(trans='log10') + geom_boxplot() + theme_bw()

ggsave(vsglint_precision_plot, file="plots/vsglint_precision_plot.pdf", width=3, height=4)
ggsave(vsglint_recall_plot, file="plots/vsglint_recall_plot.pdf", width=3, height=4)
ggsave(vsglint_time_taken_plot, file="plots/vsglint_time_plot.pdf", width=3, height=4)

adcal_precision_plot = ggplot(adcal_dat, aes(x=function_used, y=precision)) + geom_boxplot() + theme_bw()
adcal_recall_plot = ggplot(adcal_dat, aes(x=function_used, y=recall)) + geom_boxplot() + theme_bw()
adcal_time_taken_plot = ggplot(adcal_dat, aes(x=function_used, y=time_taken)) + scale_y_continuous(trans='log10') + geom_boxplot() + theme_bw()

ggsave(adcal_precision_plot,  file="plots/small_adcal_precision_plot.pdf", width=3, height=4)
ggsave(adcal_recall_plot,     file="plots/small_adcal_recall_plot.pdf", width=3, height=4)
ggsave(adcal_time_taken_plot, file="plots/small_adcal_time_plot.pdf", width=3, height=4)

large_adcal_limit_nb_precision_plot  = ggplot(large_adcal_limit_nb_dat, aes(x=function_used, y=precision)) + geom_boxplot() + theme_bw()
large_adcal_limit_nb_recall_plot     = ggplot(large_adcal_limit_nb_dat, aes(x=function_used, y=recall)) + geom_boxplot() + theme_bw()
large_adcal_limit_nb_time_taken_plot = ggplot(large_adcal_limit_nb_dat, aes(x=function_used, y=time_taken)) + scale_y_continuous(trans='log10') + geom_boxplot() + theme_bw()

ggsave(large_adcal_limit_nb_precision_plot,  file="plots/large_adcal_limit_nb_precision_plot.pdf", width=3, height=4)
ggsave(large_adcal_limit_nb_recall_plot,     file="plots/large_adcal_limit_nb_recall_plot.pdf", width=3, height=4)
ggsave(large_adcal_limit_nb_time_taken_plot, file="plots/large_adcal_limit_nb_time_plot.pdf", width=3, height=4)
