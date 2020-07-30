#!/usr/bin/env Rscript
require(ggplot2)
require(dplyr)
require(RColorBrewer)
#source("~/Projects/R/fs_.R")
#setwd("~/Projects/epistasis/results/simulation")
setwd("..")

args <- commandArgs(trailingOnly = TRUE)


if (length(args) >= 4) {
    append_str = args[4]
} else {
    append_str = ''
}

expect_adcal = FALSE
if (args[3] == 'y') {
    expect_adcal = TRUE
}

beta_limit=-1
if (args[4] == 'limit') {
    beta_limit = 2000
}

if (args[2] == 'l') {
    cat("using large data\n")
    large_int <- TRUE
    graph_numrows <- c(10000)
    graph_nbi <- c("0", "200", "500", "1000")
    graph_nbij <- c("50", "200", "500", "1000")
} else if (args[2] == 'm') {
    cat("using mixed data\n")
    graph_numrows <- c(10000)
    graph_nbi <- c("0", "20", "50", "100")
    graph_nbij <- c("5", "20", "50", "100")
    large_int <- FALSE
} else {
    cat("using small data\n")
    graph_numrows <- c(1000)
    graph_nbi <- c("0", "20", "50", "100")
    graph_nbij <- c("5", "20", "50", "100")
    large_int <- FALSE
}

fits_path=sprintf("./fits_testing")

rds_file = sprintf("PrecRecF1/dat_precrecf1_lasso")

if (args[1] == 'y') {
    # Precision, recall and F1 for interaction terms
     ans <- lapply(list.files(path = fits_path, pattern = "", full.names = TRUE), function(f) {#, sprintf("n%d_p%d", n, p)), function(f) {
       ans <- readRDS(f)
       n <- regmatches(x = f, m = regexpr(f, pattern = "(?<=n)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       p <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_p)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       SNR <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_SNR)\\d+(?=_)", perl = TRUE)) %>% as.numeric
        L <- 0
       nbi <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_nbi)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       nbij <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_nbij)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       nlethals <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_nlethals)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       perc_viol <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_viol)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       adcal <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_adcal)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       nbeta_limit <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_nbetalimit)\\d+(?=_)", perl = TRUE)) %>% as.numeric
       ID <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_)\\d+(?=_|\\.rds)", perl = TRUE)) %>% as.numeric
       smry_int <- ans$smry %>% filter(type == "interaction")
       time_taken <- ans$time[3]
       notest <- data.frame(n = n, p = p, SNR = SNR, L = L, nbi = nbi, nbij = nbij, nlethals = nlethals, time_taken = time_taken,
                            precision = sum(smry_int[["TP"]]) / nrow(smry_int),
                            recall = sum(smry_int[["TP"]]) / nbij) %>%
         mutate(F1 = 2 *  (precision * recall) / (precision + recall),
                test = "no")
       smry_int <- mutate(smry_int, pval = p.adjust(pval, method = "BH")) %>%
         filter(pval < 0.05)
       test <- data.frame(n = n, p = p, SNR = SNR, L = L, nbi = nbi, nbij = nbij, nlethals = nlethals, time_taken = time_taken,
                          precision = sum(smry_int[["TP"]]) / nrow(smry_int),
                          recall = sum(smry_int[["TP"]]) / nbij) %>%
         mutate(F1 = 2 *  (precision * recall) / (precision + recall),
                test = "yes")
       rbind(notest, test) %>% data.frame(., id = ID)
     }) %>% do.call("rbind", .) %>%
       tbl_df %>%
       mutate(n = factor(n),
              p = factor(p),
              SNR = factor(SNR),
              L = factor(L),
              nbi = factor(nbi),
              nbij = factor(nbij),
              nlethals = factor(nlethals))
     saveRDS(ans, file = rds_file)
}


for (numrows in graph_numrows) { #400
  for (t in c("yes", "no")) {
    dat_precrecf1 <- readRDS(file = rds_file)
      dat_precrecf1 <- dat_precrecf1 %>%
      filter(adcal == expect_adcal) %>%
      filter(nbeta_limit == beta_limit) %>%
      filter(n == numrows) %>%
      filter(test == t) %>%
      filter(nbi %in% graph_nbi) %>%
      filter(nbij %in% graph_nbij) %>%
      filter(SNR != "1") %>%
      filter(nlethals == 0) %>%
      mutate(F1 = case_when(is.na(F1) ~ 0, TRUE ~ F1)) %>%
      mutate(SNR = factor(SNR, levels = levels(SNR), labels = paste0("SNR = ", levels(SNR)))) %>%
      group_by(n, p, SNR, nbi, nbij, test) %>% sample_n(10, replace=TRUE) #TODO: improve this?
    
    pl <- group_by(dat_precrecf1, SNR, nbi, nbij) %>%
      summarise(mean = mean(precision, na.rm = TRUE), sd = sd(precision, na.rm = TRUE) / sqrt(n())) %>%
      mutate(Measure = "Precision") %>%
      rbind(., group_by(dat_precrecf1, SNR, nbi, nbij) %>%
              summarise(mean = mean(recall, na.rm = TRUE), sd = sd(recall, na.rm = TRUE) / sqrt(n())) %>%
              mutate(Measure = "Recall")) %>%
      rbind(., group_by(dat_precrecf1, SNR, nbi, nbij) %>%
              summarise(mean = mean(F1, na.rm = TRUE), sd = sd(F1, na.rm = TRUE) / sqrt(n())) %>%
              mutate(Measure = "F1")) %>%
      mutate(Measure = factor(Measure, levels = c("Precision", "Recall", "F1"))) %>%
      ggplot(aes(x = nbij, 
                 y = mean, 
                 group = nbi,
                 ymax = mean + sd, 
                 ymin = mean - sd)) +
      facet_grid(Measure ~ SNR) +
      # geom_bar(aes(fill = nbi), stat = "identity", position = "dodge") +
      geom_line(aes(colour = nbi), position=position_dodge(.35)) +
      geom_point(aes(colour = nbi), position=position_dodge(.35), size = 1) +
      geom_errorbar(colour = "darkgrey", width=.3, position=position_dodge(.35)) +
      scale_color_discrete(name = "True additional\nmain effects") +
      # scale_colour_manual(name = "Additional main effects", values = c("#7fcdbb", "#1d91c0", "#253494")) + #brewer.pal(7, "Set3")[-1]) + #
      ylim(c(0,1)) +
      xlab("True interactions") +
      ylab("") +
      #theme_fs() +
      theme(legend.position = "bottom")
    pl
    ggsave(pl, file = sprintf("PrecRecF1/PrecRecF1_n%d_t%s_large%d_%s.pdf", numrows, t, large_int, append_str), width = 5, height = 7)
  }
}








#
#
#
require(tidyr)
require(gridExtra)
#source("~/Projects/R/fs_.R")
for (numrows in graph_numrows) { #400
  
  dat <- readRDS(file = rds_file) %>%
    filter(n == numrows) %>%
    mutate(test = factor(test)) %>%
    group_by(n, p, SNR, nbi, nbij, test) %>%
    summarise(precision = mean(precision, na.rm = TRUE), recall = mean(recall, na.rm = TRUE)) %>%
    gather(measure, value, precision:recall) %>%
    ungroup %>%
    spread(test, value) %>%
    filter(SNR == 5) %>%
    filter(nbi %in% graph_nbi) %>% # These numbers were unquoted for some reason, this might matter.
    filter(nbij %in% graph_nbij)
  
  
  pl.prec <- group_by(subset(dat, measure == "precision"), n, p, nbi, nbij, measure) %>%
    summarise(change = yes - no) %>%
    ggplot(aes(x = nbij, y = change)) +
    geom_bar(aes(fill = nbi), stat = "identity", position = "dodge") +
    # facet_wrap(~measure, scale = "free_y", ncol = 1) +
    # scale_fill_manual(name = "Additional\nmain effects", values = c("#7fcdbb", "#1d91c0", "#253494")) +
    #scale_fill_discrete(name = "True additional\nmain effects") +
    ylim(c(0, 0.8)) +
    #xlab("True interactions") +
    ylab("Precision") +
    theme(axis.title.x=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank()) +
    #theme_fs() +
    theme(legend.position = "none")
  
  pl.rec <- group_by(subset(dat, measure == "recall"), n, p, nbi, nbij, measure) %>%
    summarise(change = yes - no) %>%
    ggplot(aes(x = nbij, y = change)) +
    geom_bar(aes(fill = nbi), stat = "identity", position = "dodge") +
    scale_y_reverse() +
    # facet_wrap(~measure, scale = "free_y", ncol = 1) +
    # scale_fill_manual(name = "Additional\nmain effects", values = c("#7fcdbb", "#1d91c0", "#253494")) +
    scale_fill_discrete(name = "True additional\nmain effects") +
    ylim(c(-0.8, 0)) +
    xlab("True interactions") +
    ylab("Recall") +
    #theme_fs() +
    theme(legend.position = "bottom")
  
  pdf(sprintf("PrecRecF1/test_analysis_n%d_large%d_xyz%s_%s.pdf", numrows, large_int, use_xyz, append_str), width = 5, height = 5)
  grid.arrange(pl.prec, pl.rec, ncol = 1)
  dev.off()
}
