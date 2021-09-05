# performs feature extraction on the datasets using variance threshold method
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from os import chdir
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from ML import *


def filter_method(X, y, cols):
    """
    Performs a chi squared test in order to find significant features (m(i)RNAs)
    :param X: (DataFrame) Contains samples against features
    :param y: (DataFrame) Targets for X
    :param cols: (list) All features
    :return:
        X_new (DataFrame) Contains significant features
    """
    # chi squared
    print(X.shape)
    X_proc = X
    chi = SelectPercentile(chi2, percentile=5)
    X_new = chi.fit_transform(X_proc, y)
    m = chi.get_support(indices=False)
    selected_columns = cols[m]
    X_new = pd.DataFrame(X_new)
    X_new.columns = selected_columns
    print(X_new.shape)
    return X_new


##########################################################################
# Recursive feature selection


def recursive_ft(X, y, pickle_name, RNA_type, min_features_to_select):
    """
    Performs recursive feature elimination on the given data and saves the RFECV object as a pickle object in the same
    location as where the data was retrieved
    :param X: (DataFrame) containing filtered features
    :param y: (DataFrame) targets
    :param pickle_name: str
    :param RNA_type: (str) miRNA or mRNA
    :param min_features_to_select: (int) 1% of total genes

    """
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications

    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select)

    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    pickle_name = pickle_name + ".pickle"
    with open(pickle_name, 'wb') as rfecv_f:
        pickle.dump(rfecv, rfecv_f)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)

    loc = r"../../../Figures/"
    name = loc + "finding_optimal_features_" + RNA_type + ".png"
    plt.savefig(name)

    # plt.show()

    dset = pd.DataFrame()
    X1 = pd.DataFrame(X)
    dset['attr'] = X1.columns
    # drop the unnecessary "samples" header
    # dset = dset.drop(columns = "Samples")

    dset['importance'] = rfecv.ranking_

    dset = dset.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)

    name = loc + "relative_importance_" + RNA_type + ".png"
    plt.savefig(name)
    # plt.show()


def continue_from_rfecv(file_name, RNA_type, X, min_features_to_select, loc):
    """
    Use this method in testing, and you do not want to regenerate the RFECV files. Continues as above, evaluating the
    RFECV and generating figures to be saved in loc

    :param file_name: (str) filename of pickle file
    :param RNA_type: (str) miRNA or mRNA
    :param X: (DataFrame) containing samples
    :param min_features_to_select: int
    :param loc: (str) dir of where to save figures
    """
    with open(file_name, 'rb') as f:
        rfecv = pickle.load(f)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)

    name = loc + "finding_optimal_features_" + RNA_type + ".png"
    plt.savefig(name)
    print("saved features against cross validation scores:", name)

    dset = pd.DataFrame()
    X1 = pd.DataFrame(X)
    dset['attr'] = X1.columns
    # drop the unnecessary "samples" header
    # dset = dset.drop(columns="Samples")

    dset['importance'] = rfecv.ranking_

    dset = dset.sort_values(by='importance', ascending=False)

    # optional for neater more legible figures
    # dset = dset.head(n=rfecv.n_features_*3)
    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)

    name = loc + "relative_importance_" + RNA_type + ".png"
    plt.savefig(name)
    print("saved feature importance: ", name)
    # plt.show()

    important_features = dset.head(n=rfecv.n_features_).attr
    filtered = X[important_features]
    print(rfecv.n_features_)
    return filtered, important_features


def tidy_data(filtered, original, conditions):
    """
    Formats the data in a neater format, attaching the phenotypes onto the filtered data

    :param filtered: DataFrame of the filtered data
    :param original: DataFrame of the original dataset
    :param conditions: DataFrame of the conditions (HD/WT)
    """
    # join names from original onto filtered

    samples = original.Samples
    s = pd.DataFrame({"Samples": samples})

    # filtered = filtered.iloc[: , 1:]
    named = filtered.join(samples)
    combined = named.join(conditions)
    return combined


###########################################################
# calling

def run():
    """
    This method identifies potentially useful biomarkers to predict HD
    The data is initially scaled using a Min-Max Scaler

    Significant biomarkers are identified in a two fold process
        - a chi squared test identifies potentially important biomarkers (low computation efforts)
        - recursive feature elimination
            - this is a powerful method which can be overly critical
            - a minimum of 1% of the total biomarkers must be selected using this method
                - allowing more scope for further training

    """

    loc = r"../../../FilteredData/"

    dir = "C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Preprocessed_Data\\test_train_splits\\outliers\\"
    chdir(dir)

    dirs = ["outliers/", "no_outliers/"]
    for d in dirs:
        chdir(dir + "..\\" + d)
        for filename in glob.glob('*test.csv'):
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                print("\nBeginning feature selection on", filename.replace(".csv", ""), "with", d.replace("/", ""))
                name = filename.replace(".csv", "").replace("_test", "")
                rna = pd.read_csv(f)
                X, y = get_X_y(rna)
                scaler = MinMaxScaler()
                scaler.fit(X.drop(columns="Samples"))
                X = scaler.transform(X.drop(columns="Samples"))
                X = pd.DataFrame(X)
                cols = rna.columns.drop(["Unnamed: 0", "Samples", "Conditions"])
                X_new = filter_method(X, y, cols)

                # select at minimum 1% of biomarkers for further investigation
                min_features_to_select = [round(0.01 * (len(X.columns))) if round(0.01 * (len(X.columns))) > 1 else 1][
                    0]

                # allow rcfev to be more powerful
                # min_features_to_select = 1
                #recursive_ft(X_new, y, (name + "_rfe"), name, min_features_to_select)

                fig_save_loc = r"../../../Figures/" + d
                RNA_data, RNAs = continue_from_rfecv((name + "_rfe.pickle"), name, X_new, min_features_to_select,
                                                     fig_save_loc)

                X["Samples"] = rna["Samples"]

                RNAs_comb = tidy_data(RNA_data, X, y)
                RNAs_comb.to_csv((loc + d + name + "_train.csv"))

                RNA_val = pd.read_csv(name + "_test.csv")
                RNA_val_filtered = RNA_val[RNAs]

                mRNA_validation = pd.read_csv(name + "_test.csv")
                X, y = get_X_y(mRNA_validation)
                mRNAs_vals_comb = tidy_data(RNA_val_filtered, X, y)
                mRNAs_vals_comb.to_csv((loc + d+ name + "_validation.csv"))

                print("saved", name, "in", loc + d)


if __name__ == "__main__":
    run()
############################################################

"""
THIS IS NOW ARCHIVED - THIS CODE IS NOT IMPLEMENTED IN PACKAGE - YOU CAN REIMPLEMENT VARIANCE THRESHOLDS BY RUNNING THE METHOD BELOW

Variance Threshold Preamble: 
I have standardised the variances by diving by the means, in order to reduce skew

From towardsdatascience.com -> https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
"It is not fair to compare the variance of a feature to another. The reason is that as the values in the distribution get bigger, the variance grows exponentially. In other words, the variances will not be on the same scale. 

One method we can use is normalizing all features by dividing them by their mean. This method ensures that all variances are on the same scale. "

"""


def selectFeatures(RNA_train_data, train_file_name, RNA_validatation_data, validate_file_name):
    # transposes the data so that the RNAs are columns
    labelled_RNAs_train = RNA_train_data.transpose()
    labelled_RNAs_validate = RNA_validatation_data.transpose()

    # sets mRNAs as columns
    labelled_RNAs_train, labelled_RNAs_train.columns = labelled_RNAs_train[1:], labelled_RNAs_train.iloc[0]
    labelled_RNAs_validate, labelled_RNAs_validate.columns = labelled_RNAs_validate[1:], labelled_RNAs_validate.iloc[0]

    RNA_counts_train = labelled_RNAs_train
    RNA_counts_validate = labelled_RNAs_validate

    print(RNA_counts_train.shape)

    # removes any RNAs where all counts are 0
    # zero_map =  (RNA_counts_train != 0).all()
    # RNA_counts_validate = RNA_counts_validate[zero_map[1]]
    # RNA_counts_train1 = RNA_counts_train[zero_map[1]]

    RNA_counts_train_zeroed = RNA_counts_train.loc[:, (RNA_counts_train != 0).any(axis=0)]
    RNA_counts_validate_zeroed = RNA_counts_validate.loc[:, (RNA_counts_train != 0).any(axis=0)]

    # print(RNA_counts_train == RNA_counts_train1)
    # match the removed columns here
    print(RNA_counts_train.shape)

    normalized_df_train = RNA_counts_train_zeroed / RNA_counts_train_zeroed.mean()

    """
    change this for increased / decreased sensitivity
    also consider recusive ft selection
    
    this is the max threshold for our raw data is 107, after about 80, the features stop tailing off and plataeu suggesting that these features are significant. the issue, is this leaves us with many columns of zeros...
    """
    vt = VarianceThreshold(threshold=107)

    # Fit
    _ = vt.fit(normalized_df_train)

    # Get the mask
    mask = vt.get_support()

    # Subset the DataFrame
    RNA_final_train = RNA_counts_train_zeroed.loc[:, mask]
    RNA_final_validate = RNA_counts_validate_zeroed.loc[:, mask]

    print(RNA_final_train.shape)

    """consider using alternative, compound feature selection including recursive ft selection to further fine tune 
    the model 
    
    """

    RNA_train_t = RNA_final_train.transpose()
    RNA_validate_t = RNA_final_validate.transpose()

    RNA_train_t.to_csv("../Early Detection/Data/" + train_file_name)
    RNA_validate_t.to_csv("../Early Detection/Data/" + validate_file_name)

    print("saved files", train_file_name, validate_file_name)
