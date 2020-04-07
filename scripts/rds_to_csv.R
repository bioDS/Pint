#!/usr/bin/Rscript

# Takes two arguments, 1: input .rds, 2: output dir.

args = commandArgs(trailingOnly=TRUE)

input_file = args[1]
output_dir = args[2]

base_filename = sub("\\..*", "", sub(".*/", "", input_file))
data = readRDS(input_file)

output_file_path = paste(sep="", output_dir, "/", base_filename, "/")
dir.create(output_file_path)

print(paste("writing to ", paste(output_file_path, sep="", "X.csv")))
write.table(data$X, file=paste(output_file_path, sep="", "X.csv"), row.names=T, col.names=F, sep=',')

print(paste("writing to ", paste(output_file_path, sep="", "Y.csv")))
write.table(data$Y, file=paste(output_file_path, sep="", "Y.csv"), row.names=T, col.names=F, sep=',')