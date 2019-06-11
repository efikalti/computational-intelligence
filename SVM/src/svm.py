from sklearn import svm, metrics
from sklearn.svm import LinearSVC
import logging
import numpy as np
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
from decomposition import Decomposition

class SVM:

    """
    Constructor
    Initialize arrays for C, gamma, class weights and decision function variables
    """
    def __init__(self):
        # Iteration variables
        self.C = [1, 10, 100, 1000]
        self.gammas = [0.001, 0.01, 0.1, 1]
        self.class_weights = [None, 'balanced']
        # Create results file
        self.results = None
        # Data for KPCA-LDA iterations
        self.dim_red = { 'kernel': '-', 'gamma': '-', 'n_component': '-' }
        # Decomposition object
        self.dec = None
        # Flag for starctaft dataset or mnist
        self.starcraft = True


    """
    Train SVM model
    If decomposition is true it will also apply KPCA-LDA to the training data with
    variables
    """
    def train(self, x_train, y_train, x_test, y_test, binary=True, iterate=False, linear=True, decomposition=False, once=False, fileprefix=''):
        self.setupResults()

        if decomposition is True:
            if once is True:
                # Create decomposition object if it does not exists otherwise use the stored object
                if self.dec is None:
                    self.dec = Decomposition()
                    x_train = self.dec.fit(x_train, y_train, kernel='rbf', gamma=0.1, n_components=10)
                else:
                    x_train = self.dec.transform(x_train)
                if self.starcraft is True:
                    self.dec.visualize(x_train, y_train, fileprefix=fileprefix)
                x_test = self.dec.transform(x_test)
                self.dim_red = { 'kernel': 'rbf', 'gamma': 0.1, 'n_component': 15 }
                self.svm(x_train, y_train, x_test, y_test, binary, iterate, linear)
            else:
                # Test with all available variables
                dec = Decomposition()
                for n_component in dec.components:
                    for kernel in dec.kernel:
                        if kernel == 'rbf':
                            for gamma in dec.Gamma:
                                x_train = dec.fit(x_train, y_train, kernel=kernel, gamma=gamma, n_components=n_component)
                                x_test = dec.transform(x_test)
                                self.dim_red = { 'kernel': kernel, 'gamma': gamma, 'n_component': n_component }
                                self.svm(x_train, y_train, x_test, y_test, binary, iterate, linear)
                        else:
                            x_train = dec.fit(x_train, y_train, n_components=n_component)
                            x_test = dec.transform(x_test)
                            self.dim_red = { 'kernel': kernel, 'gamma': '-', 'n_component': n_component }
                            self.svm(x_train, y_train, x_test, y_test, binary, iterate, linear)
        else:
            # Run without dimesionality reduction
            self.dim_red = { 'kernel': '-', 'gamma': '-', 'n_component': '-' }
            self.svm(x_train, y_train, x_test, y_test, binary, iterate, linear)


    """
    Run SVM model with x_train = data and y_train = labels and test with x_test and y_test accordingly
    By default it uses the binary svm classification
    If iterate is True it will test with all the available variables for this type of classification
    """
    def svm(self, x_train, y_train, x_test, y_test, binary, iterate, linear):
        # Single iteration
        if iterate is False:
            if binary is True:
                # Train SVM with rbf and default values for the given data
                # By default C=1.0, gamma=0.01 and class weight = None
                clf = None
                clf = svm.SVC(kernel='rbf', C=1.0, gamma=0.01)
                # Train created model
                clf.fit(x_train, y_train)
                # Predict on test data
                prediction = None
                prediction = clf.predict(x_test)
                # Log results
                self.logBinaryResults(y_test, prediction, n_support=clf.n_support_, C=1.0, gamma=0.01)
            else:
                if (linear is True):
                    # Train svm for multiclass prediction using the liblinear library implementation
                    # LinearSVC with the following parameters
                    # multiclass=ovr (one v rest) creates (n_labels) number of classifiers, another argument could be crammer_singer
                    # class_weight = balanced by default in multi class
                    clf = None
                    clf = LinearSVC(multi_class='ovr', random_state=0, class_weight='balanced')
                    # Train created model
                    clf.fit(x_train, y_train)
                    # Predict on test data
                    prediction = None
                    prediction = clf.predict(x_test)
                    # Log results
                    self.logMultiClassResults(y_test, prediction, plotConfMat=True,
                        decision_function='ovr',
                        cnfTitle='Liblinear SVM',
                        cnfFilename='_liblinear', kernel='linear')
                else:
                    # Train svm for multiclass prediction using the libsvm library implementation
                    # Do one model for balanced weights and one without
                    # libsvm SVC with the following parameters
                    # multiclass=ovr (one v rest) creates (n_labels) number of classifiers, another argument could be ovo
                    # class_weight = balanced by default in multi class
                    clf = None
                    clf = svm.SVC(kernel='rbf', gamma='auto', decision_function_shape='ovo', class_weight='balanced')
                    # Train created model
                    clf.fit(x_train, y_train)
                    # Predict on test data
                    prediction = None
                    prediction = clf.predict(x_test)
                    # Log results
                    self.logMultiClassResults(y_test, prediction, plotConfMat=True,
                            n_support=clf.n_support_,
                            cnfTitle='Libsvm SVM with Balanced weights',
                            C=1,
                            gamma='auto',
                            cnfFilename='_libsvm_balanced')

                    clf = None
                    clf = svm.SVC(kernel='rbf', gamma='auto', decision_function_shape='ovo')
                    # Train created model
                    clf.fit(x_train, y_train)
                    # Predict on test data
                    prediction = None
                    prediction = clf.predict(x_test)
                    # Log results
                    self.logMultiClassResults(y_test, prediction,
                        n_support=clf.n_support_,
                        weights='unbalanced',
                        C=1,
                        gamma='auto',
                        plotConfMat=True, cnfTitle='Libsvm SVM without weights',
                        cnfFilename='_libsvm')
        # Multiple iterations
        else:
            if binary is True:
                for c in self.C:
                    # Test binary svm with all available c values
                    for gamma in self.gammas:
                        # Test with all available gamma values
                        for weight in self.class_weights:
                            # Test with balanced and equal weights
                            clf = None
                            clf = svm.SVC(kernel='rbf', C=c, gamma=gamma, class_weight=weight)
                            # Train created model
                            clf.fit(x_train, y_train)
                            # Predict on test data
                            prediction = None
                            prediction = clf.predict(x_test)
                            # Calculate and save results
                            results = metrics.precision_recall_fscore_support(y_test, prediction, average='macro')
                            # Log results of current run
                            self.logBinaryResults(y_test, prediction, n_support=clf.n_support_, C=c, gamma=gamma, weight=weight)
            else:
                for c in self.C:
                    # Test binary svm with all available c values
                    for gamma in self.gammas:
                        # Test with all available gamma values
                        clf = None
                        clf = svm.SVC(kernel='rbf', C=c, gamma=gamma, class_weight='balanced')
                        # Train created model
                        clf.fit(x_train, y_train)
                        # Predict on test data
                        prediction = None
                        prediction = clf.predict(x_test)
                        # Calculate and save results
                        results = metrics.precision_recall_fscore_support(y_test, prediction, average='macro')
                        # Log results of current run
                        self.logMultiClassResults(y_test, prediction, C=c, gamma=gamma, decision_function='ovo')


    """
    Log binary SVM results
    """
    def logBinaryResults(self, y_test, prediction, kernel='rbf', C=1.0, gamma=1.0, n_support='-', weight='-'):
        confusionMatrix = metrics.confusion_matrix(y_test, prediction)
        result = metrics.precision_recall_fscore_support(y_test, prediction, average='macro')
        if self.starcraft is True:
            self.results = self.results.append({ 'Kernel': kernel, 'C': str(C), 'Gamma': str(gamma),
                          'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                          'Support vectors': n_support, 'Class weights': weight, 'Casual': confusionMatrix[0],
                          'Hardcore': confusionMatrix[1]}, ignore_index=True)
        else:
            self.results = self.results.append({ 'Kernel': kernel, 'C': str(C), 'Gamma': str(gamma),
                          'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                          'Support vectors': n_support, 'Class weights': weight, 'Even': confusionMatrix[0],
                          'Odd': confusionMatrix[1]}, ignore_index=True)

    """
    Log multi-class SVM results
    """
    def logMultiClassResults(self, y_test, prediction, kernel='rbf', C='-', gamma='-', weights='balanced', decision_function='ovo', n_support='-', plotConfMat=False, cnfTitle='', cnfFilename=''):
        # Calculate and save in dataframe
        confusionMatrix = metrics.confusion_matrix(y_test, prediction)
        if plotConfMat is True:
            self.plotHeatMap(confusionMatrix, 'Multi Class Heatmap ' + cnfTitle, 'multi-class' + cnfFilename)
        result = metrics.precision_recall_fscore_support(y_test, prediction, average='macro')
        if self.starcraft is True:
            self.results = self.results.append({ 'Kernel': kernel, 'C': str(C), 'Gamma': str(gamma),
                          'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                          'Support vectors': n_support, 'Class weights': weights, 'Decision Function': decision_function,
                          'KPCA-LDA Components': self.dim_red['n_component'], 'KPCA-LDA Kernel': self.dim_red['kernel'], 'KPCA-LDA Gamma': self.dim_red['gamma'],
                          'Bronze': confusionMatrix[0][0], 'Silver': confusionMatrix[1][1], 'Gold': confusionMatrix[2][2], 'Platinum': confusionMatrix[3][3],
                          'Diamond': confusionMatrix[4][4], 'Master': confusionMatrix[5][5], 'GrandMaster': confusionMatrix[6][6],
                          'Professional': confusionMatrix[7][7]}, ignore_index=True)
        else:
            self.results = self.results.append({ 'Kernel': kernel, 'C': str(C), 'Gamma': str(gamma),
                          'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                          'Support vectors': n_support, 'Class weights': weights, 'Decision Function': decision_function,
                          'KPCA-LDA Components': self.dim_red['n_component'], 'KPCA-LDA Kernel': self.dim_red['kernel'], 'KPCA-LDA Gamma': self.dim_red['gamma'],
                          'Zero': confusionMatrix[0][0], 'One': confusionMatrix[1][1], 'Two': confusionMatrix[2][2], 'Three': confusionMatrix[3][3],
                          'Four': confusionMatrix[4][4], 'Five': confusionMatrix[5][5], 'Six': confusionMatrix[6][6],
                          'Seven': confusionMatrix[7][7], 'Eight': confusionMatrix[8][8], 'Nine': confusionMatrix[9][9]}, ignore_index=True)

    """
    Plot Confusion Matrix
    """
    def plotHeatMap(self, confusionMatrix, title, filename):
        pyplot.figure(figsize=[13, 6])
        # Calculate confusion matrix
        ax = sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="OrRd")
        # Plot confusion matrix
        pyplot.title(title)
        pyplot.xlabel('True output')
        pyplot.ylabel('Predicted output')
        pyplot.savefig('logs/heatmap_' + filename + '.png')
        pyplot.clf()


    """
    Setup results dataframe object
    """
    def setupResults(self):
        if self.starcraft is True:
            self.results = pd.DataFrame(columns=['Kernel', 'C', 'Gamma', 'Precision', 'Recall', 'F1', 'Support vectors', 'Class weights', 'Decision Function',
                                                 'Casual', 'Hardcore', 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster',
                                                 'Professional'])
        else:
            self.results = pd.DataFrame(columns=['Kernel', 'C', 'Gamma', 'Precision', 'Recall', 'F1', 'Support vectors', 'Class weights', 'Decision Function',
                                                 'Even', 'Odd', 'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                                                 'Eight', 'Nine'])


    def isMnist(self):
        self.starcraft = False
