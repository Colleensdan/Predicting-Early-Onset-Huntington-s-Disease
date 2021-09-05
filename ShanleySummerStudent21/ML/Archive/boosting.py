# evaluate gradient boosting ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from ML import *
from os import chdir
##################################################
from sklearn.ensemble import GradientBoostingClassifier
import pickle
loc = r"../InputForML/SMOTE/"
chdir(loc)

import glob


def gradient_tree_boosting():
    for filename in glob.glob('*X*'):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            X_train, X_test, y_train, y_test = open_files(f, filename)

            clf = GradientBoostingClassifier(n_estimators=100,  learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
            loc = r"../../ML/Classifiers/"
            pickle_name = loc+"gradient_tree_boosting_clf"+".pickle"
            with open(pickle_name, 'wb') as clf_f:
                pickle.dump(clf, clf_f)
            print("trained ", (filename.replace("X_train_", "").replace(".csv", ""))+"s")
            print(clf.score(X_test, y_test))

gradient_tree_boosting()
