#!/usr/bin/Rscript

# Takes two arguments, 1: input .rds, 2: output dir.

args = commandArgs(trailingOnly=TRUE)

input_file = args[1]
output_dir = args[2]

base_filename = sub("\\..*", "", sub(".*/", "", input_file))
data = readRDS(input_file)

x_csv_filename = paste(sep="", output_dir, "/", base_filename, "_X.csv")
y_csv_filename = paste(sep="", output_dir, "/", base_filename, "_y.csv")

print(paste("writing to ", x_csv_filename))
write.table(data$X, file=x_csv_filename, row.names=T, col.names=F, sep=',')

print(paste("writing to ", y_csv_filename))
write.table(data$Y, file=y_csv_filename, row.names=T, col.names=F, sep=',')