# transposes data such that columns represent unique biomarkers, and then splits data in a 80/20 ratio for later training, and feature selection


# Created by: Colle
# Created on: 07/07/2021

library(data.table)
library(factoextra)
library(dplyr)
library(tibble)
library(caTools)
library(bestNormalize)
############################################
# code extracted from DE_ML_Input_mRNA.R

#set working dir
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\InputForRFiltering\\")


getColnames <- function(RNA_df){
  #get an extra column HD, ignores sex
  HD <- sub(colnames(RNA_df), pattern = "fe", replacement = "")
  HD <- sub(HD, pattern = "male_", replacement = "")
  HD <- sub(HD, pattern = "_2m", replacement = "")
  HD <- sub(HD, pattern = "_6m", replacement = "")
  HD <- sub(HD, pattern = "_10m", replacement = "")
  HD <- sub(HD, pattern = "_1R", replacement = "")
  HD <- sub(HD, pattern = "_2R", replacement = "")
  HD <- sub(HD, pattern = "_3R", replacement = "")
  HD <- sub(HD, pattern = "_4R", replacement = "")
  HD <- sub(HD, pattern = "_5R", replacement = "")
  HD <- sub(HD, pattern = "Q20", replacement = "WT")
  HD <- sub(HD, pattern = "Q111", replacement = "HD")
  HD <- sub(HD, pattern = "Q140", replacement = "HD")
  HD <- sub(HD, pattern = "Q175", replacement = "HD")
  HD <- sub(HD, pattern = "Q80", replacement = "HD")
  HD <- sub(HD, pattern = "Q92", replacement = "HD")

  # create file linking individual to phenotype
  Samples <- colnames(RNA_df)
  Conditions <- HD
  colData <- cbind(Samples, Conditions)
  rownames(colData) <- Samples
  colData <- as.data.frame(colData)
  colData <- colData[-1,]

  if (any(is.na(colData))){
    stop("Some conditions have NA instead of phenotype. Check that all columns in the df have been parsed")
  }
  return (colData)
}


############################################
# Transform RNAs into a form ready for ML

transform_for_ml <- function(RNA_data, file_name, colData, loc)
  # data is transposed such that biomarkers are columns and each row entry represents one sample
{

  RNA_t <- transpose(RNA_data)
  RNA_t <- as.data.frame(RNA_t)
  rownames(RNA_t) <- colnames(RNA_data)
  colnames(RNA_t) <- rownames(RNA_data)
  t_RNA <- RNA_t %>%
    rownames_to_column(var = "Samples")
  t_RNA <- as.data.frame(t_RNA)
  RNA_ML <- merge(t_RNA, colData)
  colnames(RNA_ML) <- c("Samples", RNA_t[1,], "Conditions")

  # check no NAs exist in the dataframe
  if (any(is.na(RNA_ML))){
    stop("NA exists in dataframe:",file_name ,". Check that all columns in the df have been parsed in the get columns function")
  }
  f <- paste(loc,file_name,".csv", sep="")
  write.csv(RNA_ML, file=f)
  print(paste("saved ", f))
  return (RNA_ML)


}

split <- function(data, name, loc){

  require(caTools)
  set.seed(101)
  sample = sample.split(data$Conditions, SplitRatio = .8)
  train = subset(data, sample == TRUE)
  test  = subset(data, sample == FALSE)
  tr <- paste(loc, name, "_train.csv", sep="")
  write.csv(train, file=tr)
  te <- paste(loc, name, "_test.csv", sep="")
  write.csv(test, file=te)
  print(paste("saved", te))
}


################################################

setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Separated_Data\\normalized_age\\outliers")

for (dir in c("outliers", "no_outliers")){
  setwd(paste("..\\", dir, sep=""))

  miRNA_2m <- read.csv("miRNA_2m.csv")
  miRNA_6m <- read.csv("miRNA_6m.csv")
  miRNA_10m <- read.csv("miRNA_10m.csv")

  mRNA_2m <- read.csv("mRNA_2m.csv")
  mRNA_6m <- read.csv("mRNA_6m.csv")
  mRNA_10m <- read.csv("mRNA_10m.csv")

  rnas <- list(miRNA_2m, miRNA_6m, miRNA_10m, mRNA_2m, mRNA_6m, mRNA_10m)
  nrnas <- list("miRNA_2m", "miRNA_6m", "miRNA_10m", "mRNA_2m", "mRNA_6m", "mRNA_10m")


  # save dir

  loc_split <-paste(paste("..\\..\\..\\Preprocessed_Data\\test_train_splits\\", dir, sep=""), "\\", sep="")
  loc_transform <- paste(paste("..\\..\\..\\Preprocessed_Data\\", dir, sep=""), "\\", sep="")
  i <- 0
  for (rna in rnas){
    i<- i+1

    col <- getColnames(rna)
    df <- transform_for_ml(rna, nrnas[i], col, loc_transform)
    split(df, nrnas[i], loc_split)
  }

}
print("data preprocessing complete")