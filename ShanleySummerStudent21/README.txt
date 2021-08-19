Run scikit-learn == 0.23.2 in order to optimise SVM classifier



Hi Colleen, please read this file first. Also, I know you have exams ect to sort out first. This is will hopefully help as a guide when you do get started.
-----------------------

Biology
-------
Huntington's disease (HD) is a neurological disease which mostly affects middle aged individuals but in rare cases it can begin in juvenile patients.

Mutations in the HDD gene lead to an expanse in CAG/glutamine/Q repeats. Leading to disfunction and
altered gene expression. * I think reading reviews would be more helpful if you would like to know more. 

microRNA expression is also changed. This is important because microRNAs can circulate in blood and thus have the ability to be non-invasive detection biomarkers. 

----------------------

Data
----

We have 336 (RNAseq and microRNAseq) cortex samples from mice aged 2, 6 or 10 months.

Samples are from wild type | 20 CAG repeats or fewer.
Or from mutants | >20 CAG repeats. This includes 80, 92, 111, 140, or 175 repeats. 

Samples are either male or female. 

I have prevously tried training based on CAG repeat lengths and gender -- no luck. 

Based on our samples I believe the best ML question we can ask is : is the sample HD or WT?

If time allows we could explore a more complex question.

---------------------

R code
------

There are four scripts in the /RScripts folder. They should be run in the following order:

1) seperate_Val.R - separates 2 month data from the 6 and 10 month data. This is because we would like to do an early detection based model. Treating the 2 month data as a validation set will stop leakage. Also removed outliers.

2) DE_ML_INPUT_MIRNA.R - performs DE on the microRNA data (6m and 10m). miRs are significantly smaller than mRNAs so it would make sense to run this file before the mRNA DE file.
# requires R <=2.5.1 for the limma package


3) DE_ML_INPUT_MRNA.R - DE on mRNA data (6m and 10m).

4) Combine_data.R - combines the mRNAs and miRNAs found to be significantly DE in the training data (6m and 10m). Extracts these genes from the validation data (2m). We use the normalised count data for this.

---------------------

InputForRFiltering
------------------

Contains count and pheno files for the miRNA and mRNA data. I think it would be good to repeat the current analysis first.

---------------------

ML
--

Admittedly I am quite niave about ML. 
I have tried a basic ML model. A few common steps are used here:

1) training (6 and 10m) and validation (2m) data are are split into data and sample names.
2) training and validation data are scaled.
3) training data is tested on a few different algorithms
4) training data is cross validated and accuracy is contrasted to the validation data 
5) confusion matrix is plotted

I have tried feature selection methods but I don't think I did it correctly. 

--------------------
I hope this summary helps. It would be good if you went through this and see where you could add to it. Also as Daryl said before, this is based on a strict bioinformatics feature reduction method. So it would be interesting to see how a more big data approach works. 

Best,
Krutik


