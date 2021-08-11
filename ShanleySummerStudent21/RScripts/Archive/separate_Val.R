library(tidyverse)
library(testthat)


setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\InputForRFiltering")
# import data

miRNA <- read.csv("miRNA_counts.csv", row.names = 1)
mRNA <- read.csv("mRNA_counts.csv", row.names = 1)

# not dealing with phenotypes for now but code can remain for future use
Pheno_miRNA <- read.csv("miRNA_Pheno.csv", row.names = 1)
Pheno_mRNA <- read.csv("mRNA_Pheno.csv")
############################################################
# remove known outliers and mRNA-microRNA inconsistencies
miRNA <- miRNA[,-c(63, 96, 127, 145, 157)]
mRNA <- mRNA[,-c(63, 96, 127, 145, 157)]
Pheno_miRNA <- Pheno_miRNA[-c(63, 96, 127, 145, 157),]
Pheno_mRNA <- Pheno_mRNA[-c(63, 96, 127, 145, 157),]
rownames(Pheno_miRNA) <- NULL
rownames(Pheno_mRNA) <- NULL
#############################################################
# check pheno names

# This line does not run, due to different names. Since sex is not
# being analysed currently, I have changed the sex. This can be
# changed if need be later on, however, why is there the same data
# point with different sexes?
# Pheno_miRNA$Name[92] <- "male_Q20_10m"
# expect_equal(Pheno_miRNA$Name, Pheno_mRNA$Name)
# 1 difference but acceptable
#############################################################
# Remove naming specifics
HD <- sub(Pheno_miRNA$Name, pattern = "fe", replacement = "")
HD <- sub(HD, pattern = "male_", replacement = "")
HD <- sub(HD, pattern = "Q20", replacement = "WT")
HD <- sub(HD, pattern = "Q111", replacement = "HD")
HD <- sub(HD, pattern = "Q140", replacement = "HD")
HD <- sub(HD, pattern = "Q175", replacement = "HD")
HD <- sub(HD, pattern = "Q80", replacement = "HD")
HD <- sub(HD, pattern = "Q92", replacement = "HD")
Pheno_miRNA$HD <- HD
Pheno_mRNA$HD <- HD
#########################################################
# Separate 2m and 6,10m data
mRNA_train <- mRNA %>% select(-contains("2m"))
miRNA_train <- miRNA %>% select(-contains("2m"))
Pheno_mRNA_train <- Pheno_mRNA %>% filter(!grepl('2m', age))
Pheno_miRNA_train <- Pheno_miRNA %>% filter(!grepl('2m', AGE))

mRNA_val <- mRNA %>% select(contains("2m"))
miRNA_val <- miRNA %>% select(contains("2m"))
Pheno_mRNA_val <- Pheno_mRNA %>% filter(grepl('2m', age))
Pheno_miRNA_val <- Pheno_miRNA %>% filter(grepl('2m', AGE))
#########################################################
# check if equal
which(colnames(mRNA_train) %in% colnames(miRNA_train) == FALSE)


# expect_equal(colnames(mRNA_train), colnames(miRNA_train))
colnames(mRNA_train[c(57:67)]) # expected differences
colnames(miRNA_train[c(57:67)])
which(colnames(mRNA_val) %in% colnames(miRNA_val) == FALSE)

# save files

setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Separated_Data\\")

write.csv(mRNA_train, "mRNA_train.csv")
write.csv(mRNA_val,"mRNA_validation.csv")
write.csv(miRNA_train, "miRNA_train.csv")
write.csv(miRNA_val, "miRNA_validation.csv")
write.csv(Pheno_mRNA_train, "pheno_train.csv")
write.csv(Pheno_mRNA_val, "pheno_validation.csv")

print("completed separation and outlier removal")

