import os

from ML import *
from ML import SVM_classifier
from ML import naive_baiyes_classifier

class Learn:
    """
    takes in parameters model: svm, naive bayes etc, evaluate performance
    trains models, saves to docx file
    """
    def __init__(self, model = "SVM", evaluate = True):
        self.model = model
        self.evaluate = evaluate
        self.current_loc = r"C:\Users\Colle\OneDrive\Documents\Boring\2021 Summer Internship\ShanleySummerStudent21\ML"

        # keeping outliers was seen to be more effective in training useful models
        self.loc = r'C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\InputForML\\outliers\\'
        self.models = ["SVM", "NB"]

        os.chdir(self.current_loc)
        self.__validate()

        if self.model == "SVM":
            self.learn_SVM()

        if self.model == "NB":
            self.learn_NB()

    def learn_SVM(self):
        if os.getcwd() != self.loc:
            chdir(self.loc)

        for filename in glob.glob('*X*'):
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                clf, X_test, y_test, y_preds, classes, document, name, X_train, y_train = SVM_classifier.classifySVM(f, filename)

                if self.evaluate:
                    evaluate_model(y_preds, y_test, X_test, y_test, clf, classes, X_train, y_train, document=document, fname="SVM_"+name.replace("_train", ""))

    def learn_NB(self):
        if os.getcwd() != self.loc:
            chdir(self.loc)

        for filename in glob.glob('*X*'):
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                clf, X_test, y_test, y_preds, classes , document, name,X_train, y_train = naive_baiyes_classifier.classifyNB(f, filename)
                if self.evaluate:
                    evaluate_model(y_preds, y_test, X_test, y_test, clf, classes, X_train, y_train, document=document, fname="NB_"+name.replace("_train", ""))

    def __validate(self):
        if self.model not in self.models:
            raise NameError("The model {name} is not valid. Please choose from {models}".format(name = self.model, models = self.models))

        if type(self.evaluate) != bool:
            raise TypeError("Evaluate must be bool")

#svm =Learn()

Learn(model="SVM")
