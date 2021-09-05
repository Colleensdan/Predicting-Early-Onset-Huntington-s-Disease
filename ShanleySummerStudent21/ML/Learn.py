from ML import *
from ML import SVM_classifier
from ML import naive_baiyes_classifier

class Learn:
    """
    Please call this class to train your desired model and evaluate its performance. Will save files to a docx file by
    default in ML/Classifiers

    Attributes:
        :param model: (str) Either SVM or NB is acceptable. SVM by default
        :param evaluate: (bool) Model performances
        will be evaluated if true. Recommended to keep true in most cases.
        :param print_scores: (bool) Will output
        the performance of the models in the console. The truth value of print_scores is only significant if evaluate
        is true. If print_scores is true, then no documents will be saved in ML/classifiers.

        :var self.current_loc (str) directory of the file Learn.py
        :var self.loc (str) directory of data to use when training models
        :var self.models (str) list of acceptable models
    """
    def __init__(self, model = "SVM", evaluate = True, print_scores=False):
        """
        Please call help(Learn) for more info
        """
        self.model = model
        self.evaluate = evaluate
        self.print_scores = print_scores
        self.current_loc = r"..\\ML"

        # keeping outliers was seen to be more effective in training useful models
        self.loc = r'..\\InputForML\\outliers\\'
        self.models = ["SVM", "NB"]

        os.chdir(self.current_loc)
        self.__validate()

        if self.model == "SVM":
            self.learn_SVM()

        if self.model == "NB":
            self.learn_NB()

    def learn_SVM(self):
        """
        Trains an optimised SVM learner, calling the SVM module.

        Attributes:
            :var f (Object - TextIOWrapper) data file to be opened
            :var clf (Object) trained SVM classifier
            :var X_test (DataFrame) A DataFrame containing individual samples per row, with biomarkers as columns. A
            subset of the data for testing model performance
            :var y_test (DataFrame) The conditions (HD/WT) of the samples in X_test
            :var y_preds (DataFrame) The conditions (HD/WT) of the data, as predicted by the model
            :var classes (List) ["HD", "WT"]
            :var document (Object) Used for parsing data from python into a docx document
            :var name (str) name of document currently working on e.g. "X_miRNA_10m_train"
            :var X_train (DataFrame) A DataFrame containing individual samples per row, with biomarkers as columns. A
            subset of the data for training model
            :var y_train (DataFrame) The conditions (HD/WT) of the samples in X_train
        """
        if os.getcwd() != self.loc:
            chdir(self.loc)

        for filename in glob.glob('*X*'):
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                clf, X_test, y_test, y_preds, classes, document, name, X_train, y_train = SVM_classifier.classifySVM(f, filename)

                if self.evaluate:
                    evaluate_model(y_preds, y_test, X_test, y_test, clf, classes, X_train, y_train, document=document, fname="SVM_"+name.replace("_train", ""), print_scores=self.print_scores)

    def learn_NB(self):
        """
        Trains a Gaussian NB learner, calling the NB module.

        Attributes:
            :var f (Object - TextIOWrapper) data file to be opened
            :var clf (Object) trained SVM classifier
            :var X_test (DataFrame) A DataFrame containing individual samples per row, with biomarkers as columns. A
            subset of the data for testing model performance
            :var y_test (DataFrame) The conditions (HD/WT) of the samples in X_test
            :var y_preds (DataFrame) The conditions (HD/WT) of the data, as predicted by the model
            :var classes (List) ["HD", "WT"]
            :var document (Object) Used for parsing data from python into a docx document
            :var name (str) name of document currently working on e.g. "X_miRNA_10m_train"
            :var X_train (DataFrame) A DataFrame containing individual samples per row, with biomarkers as columns. A
            subset of the data for training model
            :var y_train (DataFrame) The conditions (HD/WT) of the samples in X_train
        """
        if os.getcwd() != self.loc:
            chdir(self.loc)

        for filename in glob.glob('*X*'):
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                clf, X_test, y_test, y_preds, classes , document, name,X_train, y_train = naive_baiyes_classifier.classifyNB(f, filename)
                if self.evaluate:
                    evaluate_model(y_preds, y_test, X_test, y_test, clf, classes, X_train, y_train, document=document, fname="NB_"+name.replace("_train", ""), print_scores=self.print_scores)

    def __validate(self):
        """
        Private method which ensures acceptable parameters are called.
        """
        if self.model not in self.models:
            raise NameError("The model {name} is not valid. Please choose from {models}".format(name = self.model, models = self.models))

        if type(self.evaluate) != bool:
            raise TypeError("Evaluate must be bool")

#######################
# Example calls
#######################

# call an optimised SVM learner on all data saving files in ML/Classifiers
#svm =Learn()

# call a NB learner on all data not saving files, instead printing results
# this presents an abridged version of the figures
Learn(model="NB", evaluate=True, print_scores=True)
