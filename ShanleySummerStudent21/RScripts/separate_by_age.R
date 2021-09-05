library(tidyverse)
library(testthat)

setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\NormalisedData")
# import data

miRNA_w_outliers <- read.csv("normalized_miRNA_outliers.txt", row.names = 1, sep = "\t")
miRNA_no_outliers <- read.csv("normalized_miRNA_Nooutliers.txt", row.names = 1, sep = "\t")

mRNA_w_outliers <- read.csv("normalized_mRNA_outliers.txt", row.names = 1, sep="\t")
mRNA_no_outliers <- read.csv("normalized_mRNA_Nooutliers.txt", row.names = 1, sep="\t")

#########################################################
# Outliers

# Separate 2m and 6,10m data
mRNA_2m <- mRNA_w_outliers %>% select(contains("2m"))
miRNA_2m <- miRNA_w_outliers %>% select(contains("2m"))

mRNA_6m <- mRNA_w_outliers %>% select(contains("6m"))
miRNA_6m <- miRNA_w_outliers %>% select(contains("6m"))

mRNA_10m <- mRNA_w_outliers %>% select(contains("10m"))
miRNA_10m <- miRNA_w_outliers %>% select(contains("10m"))

#########################################################

# save files

setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Separated_Data\\normalized_age\\outliers")

write.csv(mRNA_2m, "mRNA_2m.csv")
write.csv(miRNA_2m,"miRNA_2m.csv")
write.csv(mRNA_6m, "mRNA_6m.csv")
write.csv(miRNA_6m, "miRNA_6m.csv")
write.csv(mRNA_10m, "mRNA_10m.csv")
write.csv(miRNA_10m, "miRNA_10m.csv")


#########################################################
# No outliers

# Separate 2m and 6,10m data
mRNA_2m <- mRNA_no_outliers %>% select(contains("2m"))
miRNA_2m <- miRNA_no_outliers %>% select(contains("2m"))

mRNA_6m <- mRNA_no_outliers %>% select(contains("6m"))
miRNA_6m <- miRNA_no_outliers %>% select(contains("6m"))

mRNA_10m <- mRNA_no_outliers %>% select(contains("10m"))
miRNA_10m <- miRNA_no_outliers %>% select(contains("10m"))

#########################################################

# save files

setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Separated_Data\\normalized_age\\no_outliers")

write.csv(mRNA_2m, "mRNA_2m.csv")
write.csv(miRNA_2m,"miRNA_2m.csv")
write.csv(mRNA_6m, "mRNA_6m.csv")
write.csv(miRNA_6m, "miRNA_6m.csv")
write.csv(mRNA_10m, "mRNA_10m.csv")
write.csv(miRNA_10m, "miRNA_10m.csv")

print("completed separation by age and outliers")


