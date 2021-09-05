"""
There are clear differences between the number of HD, and WT mice. This can be ammended using SMOTE, which artifically increases the number of samples by generating new, but similar samples. For this, I have chosen to use SVM-SMOTE which appears to be the most applicable for this situation, but this SMOTE should be adjusted later on in training, to see how this affects training results.

"""

from imblearn.over_sampling import SVMSMOTE
from pandas import read_csv
from glob import glob
from ML import *
from shutil import copy2


class SMOTE:

    def __init__(self, X=None, y=None, RNA=None, X_file_name=None, y_file_name=None, d=None, save=True):
        self.RNA = RNA
        self.X = X
        self.y = y
        self.X_file_name = X_file_name
        self.y_file_name = y_file_name
        self.d = d
        self.save = save
        self.dir = r"../Early Detection/Data/FilteredData/outliers"

        self.__validate()

    def perform_smote(self, X, y, X_fn=None, y_fn=None):
        """
        When called in isolation, keep the default save=True so SMOTE files are saved in a directory.

        This method is also used in a pipeline, where saves are unnecessary

        The data is first duplicated to ensure there are enough values present for SMOTE. The entire duplication is
        needed to prevent biases from n-samples oversampling. This effect would be particularly strong in such a small
        dataset. Duplicates are later removed.
        """
        if not self.save:
            self.fit_resample(X, y)

        # save files
        if self.save:
            if self.d is not None or X_fn is not None or y_fn is not None:

                X=X.append(X)
                y=y.append(y)

                X_resampled, y_resampled = SVMSMOTE().fit_resample(X, y)
                X_resampled = X_resampled.drop_duplicates()
                y_resampled = y_resampled.iloc[X_resampled.index]

                loc = "../../../../InputForML/" + self.d + "/{}.csv"

                X_resampled.to_csv(loc.format(X_fn))
                y_resampled.to_csv(loc.format(y_fn))
                print("saved", self.X_file_name, self.y_file_name)
                return
            raise TypeError("Please provide a directory (outliers, no_outliers) to save the SMOTE files",
                            self.X_file_name, self.y_file_name)

            return X_resampled, y_resampled

    def fit_resample(self, X, y):
        X=X.append(X)
        y=y.append(y)

        X_resampled, y_resampled = SVMSMOTE().fit_resample(X, y)
        X_resampled = X_resampled.drop_duplicates()
        y_resampled = y_resampled.iloc[X_resampled.index]


    def _prep_data(self):
        X, y = get_X_y(self.RNA)
        X = X.drop(columns="Samples")
        return X, y


    def __validate(self):
        if self.save:
            if self.X_file_name is None or self.y_file_name == None:
                raise NameError("Please provide X, y file names")

            if not type(self.X_file_name) == type(self.y_file_name) == str:
                raise TypeError("Filenames must be strings")

        # ensure data has been provided for either RNA (only) or for both X, and y
        if (self.RNA is None and (self.X is None or self.y is None)) or (self.RNA is not None and (self.X is not None or self.y is not None)):
            raise NameError("Provide either dataset containing conditions, or these separated into X, y files")

        if self.X is not None and self.y is None:
            raise TypeError("Provide values for both X, and y. y is currently None")

        if self.y is not None and self.X is None:
            raise TypeError("Provide values for both X, and y. X is currently None")

    def run(self):
        """
        Call this method when running SMOTE in isolation, to save all files used in this analysis
        """
        chdir(self.dir)

        for d in ["outliers/", "no_outliers/"]:
            chdir("../" + d)
            for filename in glob.glob('*train*'):
                with open(os.path.join(os.getcwd(), filename), 'r') as f:
                    rna = read_csv(f)
                    n = filename.replace(".csv", "")
                    self._prep_data()
                    self.perform_smote(rna, ("X_" + n), ("y_" + n), d)
            for filename in glob.glob('*validation*'):
                copy2(filename, r"../../../../InputForML/" + d)
