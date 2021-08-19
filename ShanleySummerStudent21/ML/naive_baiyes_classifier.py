from sklearn.naive_bayes import GaussianNB
from ML import *
from docx import Document


def classifyNB(f, filename, save_document=True):
    """
    classifies naive bayes learner
    :param f:
    :param filename:
    :param save_document:
    :return: returns more vars if save_doc is true, take care when calling function
    """
    clf = GaussianNB()

    X_train, X_test, y_train, y_test = get_age_files(f, filename, remove_duplicates=False)

    X_train, X_test = _removeknownfeatures(X_train, X_test, filename)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    rna = filename.replace("X_", "").replace(".csv", "")

    if save_document:
        document = Document()
        document.add_heading("Gaussian Naive Baiyes Classifier Performance on {} data".format(filename.replace("X_", "").replace("_", " ").replace("train.csv", ""), "classifier"),0)
        document.add_heading("Training", level=2)
        document.add_paragraph(('Training Accuracy : %.3f'%clf.score(X_train, y_train)))
        return clf, X_test, y_test, y_preds, ["HD", "WT"], document, rna, X_train, y_train

    print(filename,'\nTraining Accuracy : %.3f'%clf.score(X_train, y_train))
    return clf, X_test, y_test, y_preds, ["HD", "WT"], rna,X_train, y_train


def _removeknownfeatures(X_train, X_test, fname):
    if "2m" in fname:
        X_train = X_train.drop(columns=["mmu-miR-465a-3p",
                                        "mmu-miR-465c-3p",
                                        "mmu-miR-465b-3p",
                                        "mmu-miR-20b-3p",
                                        "mmu-miR-7008-5p",
                                        "mmu-miR-1982-5p",
                                        "mmu-miR-12182-3p",
                                        "mmu-miR-10a-3p"])


        X_test = X_test.drop(columns=["mmu-miR-465a-3p",
                                      "mmu-miR-465c-3p",
                                      "mmu-miR-465b-3p",
                                      "mmu-miR-20b-3p",
                                      "mmu-miR-7008-5p",
                                      "mmu-miR-1982-5p",
                                      "mmu-miR-12182-3p",
                                      "mmu-miR-10a-3p"])


        return X_train, X_test
    return X_train, X_test
"""

for filename in glob.glob('*X*'):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        clf, X_test, y_test, y_preds, classes ,document, name,X_train, y_train = classifyNB(f, filename, save_document=True)
        evaluate_model(y_preds, y_test, X_test, y_test, clf, classes, X_train, y_train, document=document, fname=name.replace("_train", ""))"""
