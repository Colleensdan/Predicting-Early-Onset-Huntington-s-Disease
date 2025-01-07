Prerequisites:
    scikit-learn == 0.23.2 in order to optimise SVM classifier

On first run to generate data:
    RScripts/
        1) separate_by_age.R
        2) preprocess_data.R
        3) feature_selection.py
        4) improve_group_balances.py

    Dataflow:
        1) Early Detection/Data/NormalisedData
            ->  Early Detection/Data/Separated_Data/normalized_age

        2) Early Detection/Data/Separated_Data/normalized_age
            ->  Early Detection/Data/Preprocessed_Data/no_outliers
            ->  Early Detection/Data/Preprocessed_Data/outliers

            ->  Early Detection/Data/Preprocessed_Data/test_train_splits/no_outliers
            ->  Early Detection/Data/Preprocessed_Data/test_train_splits/outliers

        3) Early Detection/Data/Preprocessed_Data/test_train_splits/(no_)outliers
            ->  Early Detection/Data/FilteredData/no_outliers
            ->  Early Detection/Data/FilteredData/outliers

        4) Early Detection/Data/FilteredData/(no_)outliers
            -> InputForML/no_outliers
            -> InputForML/outliers

Machine Learning
    Run Learn.py, see the bottom of the script for example running

Each module is documented, please email me if you have any issues running this/understanding the modules
    Just to let you know, I am away from the 6th-12th September so will be unlikely to reply to emails within this time


You can view the presentation at: https://www.overleaf.com/read/mjkzgrbqgqhk
Or see it attached within Resources
