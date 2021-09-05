from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from ML import *

class DecisionBoundaries:
    """
    Plots the decision boundaries on data when transformed to 2d using pca (it is impossible to visualise data in more than three dimensions, and often there are more than 3 parameters).

    This will not work when all data is from one category as seen in some mRNA data, it returns an error message in this case but continues to process the other data

    The model is default linear svm, this can be changed when calling, alongside model name for naming

    If calling plot_all, specify loc when calling

    Attributes:
        :param loc (str) directory to save figure
        :param model (obj) the model used to create these plots
        :param model_name (str) an identifier for this model for producing figures
    """
    def __init__(self, model = svm.SVC(kernel='linear'), name="linear SVC"):
        """
        Please call help(DecisionBoundaries) for more info
        """
        self.loc = ""
        self.model = model
        self.model_name = name

    def plot(self, X, y, fname):
        """
        Plots one figure based off of one aggregate (X,y) of data
        :param X: (DataFrame) containing samples and features
        :param y: (DataFrame) the known phenotypes of X
        :param fname: (str) name to save this file

        """
        le = LabelEncoder()
        pca = PCA(n_components=2)

        y = le.fit_transform(y)
        Xreduced = pca.fit_transform(X)


        try:
            clf = self.model.fit(Xreduced, y)
            fig, ax = plt.subplots()
            # title for the plots
            # Set-up grid for plotting - training.
            X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
            xx, yy = self._make_meshgrid(X0, X1)

            self._plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title('Decision surface using PCA transformed/projected features with')
            ax.legend()
            plt.savefig(fname)
            return clf


        except ValueError:
            raise ValueError("The number of classes has to be greater than one; got 1 class")

    def test_plot(self, clf, X, y, fname, X_o, y_o):
        """
        Not fully implemented yet. Partial implementation of plotting training plots alongside test plots on a
        decision surface - see note when calling

        :param clf: (obj) a trained classifier
        :param X: (DataFrame) training points
        :param y: (DataFrame) training targets
        :param fname: (str) name of file to save
        :param X_o: (DataFrame) test points
        :param y_o: (DataFrame) test targets
        :return:
        """
        le = LabelEncoder()
        pca = PCA(n_components=2)

        y = le.fit_transform(y)
        Xreduced = pca.fit_transform(X)


        try:
            fig, ax = plt.subplots()
            # title for the plots
            # Set-up grid for plotting - training.

            X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
            xx, yy = self._make_meshgrid(X0, X1)

            self._plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title('Decision surface using PCA transformed/projected features with', self.model_name)
            ax.legend()
            plt.savefig(fname)
            return clf


        except ValueError:
            print("ValueError: The number of classes has to be greater than one; got 1 class")


    def plot_all(self):
        """
        Plots all decision surfaces based off all files within the given directory (self.loc)
        """
        chdir(self.loc)
        for filename in glob.glob('*X*'):
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                X_train, X_test, y_train, y_test = get_age_files(f, filename)
                X = X_train
                y = y_train
                le = LabelEncoder()
                y = le.fit_transform(y)
                pca = PCA(n_components=2)
                Xreduced = pca.fit_transform(X)
                try:
                    clf = self.model.fit(Xreduced, y)
                    fig, ax = plt.subplots()
                    # title for the plots
                    title = ('Decision surface of', self.model_name)
                    # Set-up grid for plotting.
                    X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
                    xx, yy = self._make_meshgrid(X0, X1)

                    self._plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
                    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                    ax.set_ylabel('PC2')
                    ax.set_xlabel('PC1')
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_title('Decision surface using the PCA transformed/projected features')
                    ax.legend()
                    plt.show()
                except ValueError:
                    print("Error displaying",filename)
                    print("ValueError: The number of classes has to be greater than one; got 1 class")


    def _make_meshgrid(self,x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def _plot_contours(self,ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out



