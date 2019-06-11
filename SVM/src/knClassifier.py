from sklearn import metrics
import logging
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from decomposition import Decomposition
import pandas as pd

class KNClassifier:

    """
    Constructor
    """
    def __init__(self):
        # Values for KNN
        self.neighbors = [3, 5, 10, 50, 100]
        # Logger object set in setupLogger function
        self.multiclass = False
        # Create results file
        self.results = None
        # Data for KPCA-LDA iterations
        self.dim_red = { 'kernel': '-', 'gamma': '-', 'n_component': '-' }
        # Flag for starctaft dataset or mnist
        self.starcraft = True


    """
    Train KNearest K Nearest Neighbors Classifier and Nearest Centroid Classifier model
    with x_train = data and y_train = labels and test with x_test and y_test accordingly
    """
    def train(self, x_train, y_train, x_test, y_test, decomposition=False, visualize=False, multiclass=False, once=False):
        # Update multiclass flag and setup results
        self.multiclass = multiclass
        self.setupResults()

        if decomposition is True:
            # Create decomposition object
            dec = Decomposition()
            # Plot dimensionality reduction to 2D
            if visualize is True:
                dec.test_visualize(x_train, y_train)
            if once is True:
                x_train = dec.fit(x_train, y_train, kernel='rbf', gamma=0.1, n_components=10)
                x_test = dec.transform(x_test)
                self.dim_red = { 'kernel': 'rbf', 'gamma': 0.1, 'n_component': 10 }
                # Test for all available values for neighbors
                self.nearestNeighbor(x_train, y_train, x_test, y_test, once=once)
                # Test with Nearest Centroid
                self.nearestCentroid(x_train, y_train, x_test, y_test)
            else:
                # Test with all available variables
                for n_component in dec.components:
                    for kernel in dec.kernel:
                        if kernel != 'linear':
                            for gamma in dec.Gamma:
                                x_train = dec.fit(x_train, y_train, kernel=kernel, gamma=gamma, n_components=n_component)
                                x_test = dec.transform(x_test)
                                self.dim_red = { 'kernel': kernel, 'gamma': gamma, 'n_component': n_component }
                                # Test for all available values for neighbors
                                self.nearestNeighbor(x_train, y_train, x_test, y_test)
                                # Test with Nearest Centroid
                                self.nearestCentroid(x_train, y_train, x_test, y_test)
                        else:
                            x_train = dec.fit(x_train, y_train, n_components=n_component)
                            x_test = dec.transform(x_test)
                            self.dim_red = { 'kernel': kernel, 'gamma': '-', 'n_component': n_component }
                            # Test for all available values for neighbors
                            self.nearestNeighbor(x_train, y_train, x_test, y_test)
                            # Test with Nearest Centroid
                            self.nearestCentroid(x_train, y_train, x_test, y_test)
        else:
            # Run without dimesionality reduction
            self.dim_red = { 'kernel': '-', 'gamma': '-', 'n_component': '-' }
            # Test for all available values for neighbors
            self.nearestNeighbor(x_train, y_train, x_test, y_test, once=once)
            # Test with Nearest Centroid
            self.nearestCentroid(x_train, y_train, x_test, y_test)


    """
    Runs KNN for all available neighbor values with these data
    """
    def nearestNeighbor(self, x_train, y_train, x_test, y_test, once=False):
        if once is True:
            # Test for 10 neighbors
            clf = None
            clf = KNeighborsClassifier(n_neighbors=10)
            # Train created model
            clf.fit(x_train, y_train)
            # Predict on test data
            prediction = None
            prediction = clf.predict(x_test)
            # Log results
            self.logResults(y_test, prediction, n_neighbors=10)
        else:
            # Test for all available values for neighbors
            for neighbor in self.neighbors:
                clf = None
                clf = KNeighborsClassifier(n_neighbors=neighbor)
                # Train created model
                clf.fit(x_train, y_train)
                # Predict on test data
                prediction = None
                prediction = clf.predict(x_test)
                # Log results
                self.logResults(y_test, prediction, n_neighbors=neighbor)


    """
    Runs NC for these data
    """
    def nearestCentroid(self, x_train, y_train, x_test, y_test):
        # Test with Nearest Centroid
        clf = None
        clf = NearestCentroid(metric='euclidean')
        # Train created model
        clf.fit(x_train, y_train)
        # Predict on test data
        prediction = None
        prediction = clf.predict(x_test)
        # Log results
        self.logResults(y_test, prediction, kn=False)


    """
    Log results
    """
    def logResults(self, y_test, prediction, kn=True, n_neighbors='-'):
        confusionMatrix = metrics.confusion_matrix(y_test, prediction)
        if kn is True:
            algorithm = 'knn'
        else:
            algorithm = 'nc'
        # Calculate precision, recall, f1
        result = metrics.precision_recall_fscore_support(y_test, prediction, average='macro')
        if self.starcraft is True:
            if self.multiclass is not True:
                # Log results for binary
                self.results = self.results.append({ 'Algorithm': algorithm, '#Neighbors': n_neighbors,
                              'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                              'Casual': confusionMatrix[0], 'Hardcore': confusionMatrix[1]}, ignore_index=True)
            else:
                # Log results for multiclass
                self.results = self.results.append({ 'Algorithm': algorithm, '#Neighbors': n_neighbors,
                              'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                              'KPCA-LDA Components': self.dim_red['n_component'], 'KPCA-LDA Kernel': self.dim_red['kernel'], 'KPCA-LDA Gamma': self.dim_red['gamma'],
                              'Bronze': confusionMatrix[0][0], 'Silver': confusionMatrix[1][1], 'Gold': confusionMatrix[2][2], 'Platinum': confusionMatrix[3][3],
                              'Diamond': confusionMatrix[4][4], 'Master': confusionMatrix[5][5], 'GrandMaster': confusionMatrix[6][6],
                              'Professional': confusionMatrix[7][7]}, ignore_index=True)
        else:
            if self.multiclass is not True:
                # Log results for binary
                self.results = self.results.append({ 'Algorithm': algorithm, '#Neighbors': n_neighbors,
                              'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                              'Even': confusionMatrix[0], 'Odd': confusionMatrix[1]}, ignore_index=True)
            else:
                # Log results for multiclass
                self.results = self.results.append({ 'Algorithm': algorithm, '#Neighbors': n_neighbors,
                              'Recall': float("%0.3f"%result[1]), 'Precision':  float("%0.3f"%result[0]), 'F1': float("%0.3f"%result[2]),
                              'KPCA-LDA Components': self.dim_red['n_component'], 'KPCA-LDA Kernel': self.dim_red['kernel'], 'KPCA-LDA Gamma': self.dim_red['gamma'],
                              'Zero': confusionMatrix[0][0], 'One': confusionMatrix[1][1], 'Two': confusionMatrix[2][2], 'Three': confusionMatrix[3][3],
                              'Four': confusionMatrix[4][4], 'Five': confusionMatrix[5][5], 'Six': confusionMatrix[6][6],
                              'Seven': confusionMatrix[7][7], 'Eight': confusionMatrix[8][8], 'Nine': confusionMatrix[9][9]}, ignore_index=True)



    """
    Setup results dataframe object
    """
    def setupResults(self):
        self.results = pd.DataFrame(columns=['Algorithm', '#Neighbors', 'Precision', 'Recall', 'F1', 'Casual', 'Hardcore',
                                             'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster', 'Professional'])
    """
    Setup results dataframe object
    """
    def setupResults(self):
        if self.starcraft is True:
            self.results = pd.DataFrame(columns=['Algorithm', '#Neighbors', 'Precision', 'Recall', 'F1', 'Casual', 'Hardcore',
                                                 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster', 'Professional'])
        else:
            self.results = pd.DataFrame(columns=['Algorithm', '#Neighbors', 'Precision', 'Recall', 'F1',
                                                 'Even', 'Odd', 'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                                                 'Eight', 'Nine'])


    def isMnist(self):
        self.starcraft = False
