"""
There are clear differences between the number of HD, and WT mice. This can be ammended using SMOTE, which artifically increases the number of samples by generating new, but similar samples. For this, I have chosen to use SVM-SMOTE which appears to be the most applicable for this situation, but this SMOTE should be adjusted later on in training, to see how this affects training results.

"""

from imblearn.over_sampling import SVMSMOTE
from pandas import read_csv
from varname import nameof
from glob import glob
from ML import *



def perform_smote(RNA, X_file_name, y_file_name):
    X, y = get_X_y(RNA)
    X= X.drop(columns ="Samples")
    X_resampled, y_resampled = SVMSMOTE().fit_resample(X, y)
    X_resampled = X_resampled.drop_duplicates()
    y_resampled = y_resampled.iloc[X_resampled.index]

# save files
    loc = "../../../../InputForML/SMOTE/{}.csv"

    X_resampled.to_csv(loc.format(X_file_name))
    y_resampled.to_csv(loc.format(y_file_name))
    print("saved", X_file_name, y_file_name)

def oversample(df, n):
    # minority class is always WT
    m = df["Conditions"]
    # Non WT is transformed to NaN
    m = m.where(df["Conditions"]=="WT")
    m=m.dropna()
    # select n terms from minority class
    added = m.sample(n, replace=True)
    # add these to the dataset
    df =df.append(df.loc[added.index])
    return df.drop_duplicates

dir = r"../Early Detection/Data/FilteredData/age"
chdir(dir)
for filename in glob.glob('*train*'):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        rna = read_csv(f)
        n = filename.replace(".csv", "")

        # duplicate data so enough values present for SMOTE, duplicate rather than random oversample to avoid
        # potential biases from oversampling
        rna = rna.append(rna)

        # then duplicates are deleted here
        perform_smote(rna, ("X_"+n), ("y_"+n))

