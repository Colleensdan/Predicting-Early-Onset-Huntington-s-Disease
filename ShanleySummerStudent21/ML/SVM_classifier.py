from skopt import BayesSearchCV

from ML import *
from docx import Document


loc = r"../InputForML/SMOTE/"
chdir(loc)

def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
                         n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                         n_points=1, iid=True, refit=True, cv=None, verbose=0,
                         pre_dispatch='2*n_jobs', random_state=None,
                         error_score='raise', return_train_score=False):

    self.search_spaces = search_spaces
    self.n_iter = n_iter
    self.n_points = n_points
    self.random_state = random_state
    self.optimizer_kwargs = optimizer_kwargs
    self._check_search_space(self.search_spaces)
    self.fit_params = fit_params

    super(BayesSearchCV, self).__init__(
        estimator=estimator, scoring=scoring,
        n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
        pre_dispatch=pre_dispatch, error_score=error_score,
        return_train_score=return_train_score)

BayesSearchCV.__init__ = bayes_search_CV_init

def classifySVM(f, filename, save_document=True):
    """
    classifies svm learner
    :param f:
    :param filename:
    :param save_document:
    :return: returns more vars if save_doc is true, take care when calling function
    """
    #clf = svm.SVC(gamma=0.001, C=100., probability=True)

    X_train, X_test, y_train, y_test = get_age_files(f, filename, remove_duplicates=False)


        # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(
        svm.SVC(),
        {
            'C': (1e-6, 1e+6, 'log-uniform'),
            'gamma': (1e-6, 1e+1, 'log-uniform'),
            'degree': (1, 8),  # integer valued parameter
            'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
        },
        n_iter=32,
        cv=3
    )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    #clf.fit(X_train, y_train)
    y_preds = opt.predict(X_test)

    rna = filename.replace("X_", "").replace(".csv", "")

    if save_document:
        document = Document()
        document.add_heading("Classifier Performance on {} data".format(filename.replace("X_", "").replace("_", " ").replace("train.csv", ""), "classifier"),0)
        document.add_heading("Training", level=2)
        document.add_paragraph(('Training Accuracy : %.3f'%opt.score(X_train, y_train)))
        return opt, X_test, y_test, y_preds, ["HD", "WT"], document, rna, X_train, y_train

    print(filename,'\nTraining Accuracy : %.3f'%opt.score(X_train, y_train))
    return opt, X_test, y_test, y_preds, ["HD", "WT"], rna,X_train, y_train



""":cvar



for filename in glob.glob('*X*'):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        clf, X_test, y_test, y_preds, classes , document, name,X_train, y_train = classifySVM(f, filename)
        evaluate_model(y_preds, y_test, X_test, y_test, clf, classes, X_train, y_train, document=document, fname=name.replace("_train", ""))

"""