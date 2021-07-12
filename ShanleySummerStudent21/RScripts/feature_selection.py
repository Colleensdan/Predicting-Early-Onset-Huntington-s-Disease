# performs feature extraction on the datasets using variance threshold method


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



def filter_method(X, y):
    # chi squared
    X_proc= X.iloc[: , 1:]
    print(X_proc.shape)
    chi = SelectPercentile(chi2, percentile=5)
    X_new = chi.fit_transform(X_proc, y)
    m = chi.get_support(indices=False)
    cols = X.columns
    cols = cols[1:]
    selected_columns = cols[m]
    X_new = pd.DataFrame(X_new)
    X_new.columns = selected_columns
    print(X_new.shape)
    return X_new


##########################################################################
# Recursive feature selection



def recursive_ft(X,y, pickle_name, RNA_type):
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select)

    X_new= X.iloc[: , 1:]
    rfecv.fit(X_new, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    pickle_name = pickle_name+".pickle"
    with open(pickle_name, 'wb') as rfecv_f:
        pickle.dump(rfecv, rfecv_f)


    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)

    loc = r"../Figures/"
    name = loc+"finding_optimal_features_"+RNA_type+".png"
    plt.savefig(name)

    # plt.show()

    dset = pd.DataFrame()
    X1 = pd.DataFrame(X_new)
    dset['attr'] = X1.columns
    # drop the unnecessary "samples" header
    dset = dset.iloc[1: , :]

    dset['importance'] = rfecv.ranking_

    dset = dset.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)

    name = loc+"relative_importance_"+RNA_type+".png"
    plt.savefig(name)
    # plt.show()


def continue_from_rfecv(file_name, RNA_type, X):
    with open(file_name, 'rb') as f:
        rfecv = pickle.load(f)

    print("Optimal number of features : %d" % rfecv.n_features_)

    min_features_to_select = 1
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)

    loc = r"../Figures/"
    name = loc+"finding_optimal_features_"+RNA_type+".png"
    plt.savefig(name)

    # plt.show()

    dset = pd.DataFrame()
    X1 = pd.DataFrame(X)
    dset['attr'] = X1.columns
    # drop the unnecessary "samples" header
    dset = dset.iloc[1: , :]

    dset['importance'] = rfecv.ranking_

    dset = dset.sort_values(by='importance', ascending=False)

    # optional for neater more legible figures
    #dset = dset.head(n=rfecv.n_features_*3)
    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)

    name = loc+"relative_importance_"+RNA_type+".png"
    plt.savefig(name)
    # plt.show()

    important_features =dset.head(n=rfecv.n_features_).attr
    filtered = X[important_features]
    return filtered, important_features

def tidy_data(filtered, original, conditions):
    # join names from original onto filtered
    samples = original.Samples
    s = pd.DataFrame({"Samples": samples})
    filtered = filtered.iloc[: , 1:]
    # todo, make Samples the first column
    named = filtered.join(samples)
    combined = named.join(conditions)
    return combined
###########################################################
# calling

def run():
    dir = "C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Preprocessed_Data\\"
    chdir(dir)

    loc = r"../FilteredData/"
    #mRNA----------------------------------------------
    mRNA = pd.read_csv("mRNA_train.csv")
    X,y = get_X_y(mRNA)
    X_new = filter_method(X,y)
    # un comment this line if re-generating rfe
    # recursive_ft(X_new,y, "mRNA_rfe", "mRNA")
    mRNA_data, mRNAs = continue_from_rfecv("mRNA_rfe.pickle", "mRNA", X_new)

    mRNAs_comb = tidy_data(mRNA_data, X, y)
    mRNAs_comb.to_csv((loc+"mRNA_train.csv"))

    mRNA_val = pd.read_csv("mRNA_validation.csv")
    mRNA_val_filtered = mRNA_val[mRNAs]

    mRNA_validation = pd.read_csv("mRNA_validation.csv")
    X,y = get_X_y(mRNA_validation)
    mRNAs_vals_comb = tidy_data(mRNA_val_filtered, X, y)
    mRNAs_vals_comb.to_csv((loc+"mRNA_validation.csv"))

    #miRNA------------------------------------------
    miRNA = pd.read_csv("miRNA_train.csv")
    X,y = get_X_y(miRNA)
    X_new = filter_method(X,y)
    # un comment this line if re-generating rfe
    # recursive_ft(X_new,y, "mRNA_rfe", "mRNA")
    miRNA_data, miRNAs = continue_from_rfecv("miRNA_rfe.pickle", "miRNA", X_new)

    miRNAs_comb = tidy_data(miRNA_data, X, y)
    miRNAs_comb.to_csv((loc+"miRNA_train.csv"))

    miRNA_val = pd.read_csv("miRNA_validation.csv")
    miRNA_val_filtered = miRNA_val[miRNAs]

    miRNA_validation = pd.read_csv("miRNA_validation.csv")
    X,y = get_X_y(miRNA_validation)
    miRNAs_vals_comb = tidy_data(miRNA_val_filtered, X, y)
    miRNAs_vals_comb.to_csv((loc+"miRNA_validation.csv"))

run()
############################################################

"""
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
    #RNA_counts_validate = RNA_counts_validate[zero_map[1]]
    #RNA_counts_train1 = RNA_counts_train[zero_map[1]]

    RNA_counts_train_zeroed = RNA_counts_train.loc[:, (RNA_counts_train != 0).any(axis=0)]
    RNA_counts_validate_zeroed = RNA_counts_validate.loc[:, (RNA_counts_train != 0).any(axis=0)]

    #print(RNA_counts_train == RNA_counts_train1)
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

"""
mRNA_train_data = pd.read_csv(r"../Early Detection/Data/mRNA_train.csv")
mRNA_validate_data = pd.read_csv(r"../Early Detection/Data/mRNA_validation.csv")

miRNA_data = pd.read_csv(r"../Early Detection/Data/miRNA_train.csv")
miRNA_validate_data = pd.read_csv(r"../Early Detection/Data/miRNA_validation.csv")

selectFeatures(mRNA_train_data, "sig_mRNA_train.csv", mRNA_validate_data, "sig_mRNA_validate.csv" )
selectFeatures(miRNA_data, "sig_miRNA_train.csv", miRNA_validate_data, "sig_miRNA_validate.csv")

"""
