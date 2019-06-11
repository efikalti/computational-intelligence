import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cluster
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csgraph


class Cluster:

    """
    Constructor
    Initializes the class variables necessary for preprocessing the data
    """
    def __init__(self):
        self.lle = None
        self.n_clusters = None
        self.size = None
        self.iterations = None
        self.results = None
        self.n_vectors = 5
        self.affinities = ['rbf', 'nearest_neighbors']
        self.laplacians = ['custom', 'csgraph']
        self.eigvectors = [5, 15]
        self.clusters = [3, 5, 7, 8]
        #self.eigvectors = [5, 10, 15, 20]


    """
    Run Locally Linear Embedding and Spectral Clustering on the provided data
    LLE reduces the data to 2D
    """
    def train(self, x_train, y_train, multiple=False, binary=False):

        # Set number of clusters
        self.n_clusters = 2
        # Set the size to the training set size
        self.size = len(x_train)
        # Create list with numbers from 1 to number of training items
        self.iterations = np.zeros(self.size)
        for i in range(0, self.size):
            self.iterations[i] = i+1

        # Apply Locally Linear Embedding on training and testing data
        x_train = self.LLE(x_train)

        # Plot training data
        self.filenale_ = 'multiclass'
        if binary is True:
            self.filenale_ = 'binary'
        self.visualize2D(x_train[:, 0], x_train[:, 1], c=y_train, title='Training data ' + self.filenale_,
                         filename='logs/plots/training_data_' + self.filenale_)

        # Change y_train labels for binary
        for i in range(0, len(y_train)):
            if y_train[i] == -1:
                y_train[i] = 0

        # Run SpectralClustering
        if multiple is True:
            for affinity in self.affinities:
                for laplacian in self.laplacians:
                    for vector in self.eigvectors:
                        self.n_vectors = vector
                        if binary is True:
                            self.SpectralClustering(x_train, y_train, affinity=affinity, laplacian=laplacian)
                        else:
                            for n in self.clusters:
                                self.n_clusters = n
                                self.SpectralClustering(x_train, y_train, affinity=affinity, laplacian=laplacian)
        else:
            if binary is not True:
                self.n_clusters = 8
                self.n_vectors = 8
            self.SpectralClustering(x_train, y_train)

        if multiple is True:
            for affinity in self.affinities:
                # Run with sklearns Spectral Clustering
                sklearn_predicted = self.SklearnSP(x_train, affinity=affinity)
                title = 'SKLearn SpectralClustering Results for ' + self.filenale_ + ", " + 'affinity=' + affinity
                filename='logs/plots/' + affinity + '_sklearn_' + self.filenale_
                self.visualize2D(x_train[:, 0], x_train[:, 1], c=sklearn_predicted, title=title, filename=filename)
        else:
                # Run with sklearns Spectral Clustering
                sklearn_predicted = self.SklearnSP(x_train)
                self.logResults(y_train, sklearn_predicted, sklearn=True, affinity=affinity, laplacian=laplacian)
                title = 'SKLearn SpectralClustering Results for ' + self.filenale_ + ", " + 'affinity=rbf'
                filename='logs/plots/rbf_sklearn_' + self.filenale_
                self.visualize2D(x_train[:, 0], x_train[:, 1], c=sklearn_predicted, title=title, filename=filename)




    """
    Run Spectral Clustering for these data with these parameters
    affinity=['rbf', 'nearest_neighbors'], laplacian=['custom', 'csgraph']
    Default is nearest_neighbors kernel for similarity matrix, custom for laplacian matrix
    """
    def SpectralClustering(self, x_train, y_train, affinity='nearest_neighbors', laplacian='custom'):

        # Get similarity matrix for train data
        if affinity == 'nearest_neighbors':
            similarity_matrix = self.NNGraph(x_train)
        else:
            similarity_matrix = self.SimilarityMatrix(x_train)

        # Get laplacian matrix from similarity matrix
        if laplacian == 'csgraph':
            laplacian_matrix = csgraph.laplacian(similarity_matrix, normed=False)
        else:
            laplacian_matrix = self.LaplacianMatrix(similarity_matrix=similarity_matrix)

        # Transform data using the laplacian matrix
        transormed_data = self.transformDataToLaplacian(laplacian_matrix)

        # Cluster transormed data with kmeans
        model = cluster.KMeans(n_clusters=self.n_clusters, precompute_distances='auto', random_state=0)
        predicted = model.fit(transormed_data).labels_

        self.logResults(y_train, predicted, affinity=affinity, laplacian=laplacian)
        title = 'Custom SpectralClustering Results ' + self.filenale_ + ", " + 'affinity=' + affinity + ", laplacian=" + laplacian + ", vectors=" + str(self.n_vectors)
        filename='logs/plots/' + affinity + '_' + laplacian + "_" + str(self.n_vectors) + "_" + str(self.n_clusters) + '_custom_' + self.filenale_
        self.visualize2D(x_train[:, 0], x_train[:, 1], c=predicted, title=title, filename=filename)


    """
    Create the new data using the laplacian matrix and its eigenvalues and eigenvectors
    """
    def transformDataToLaplacian(self, laplacian_matrix):
        # Get eigenvalues and eigenvectors from the laplacian matrix
        eigval, eigvec = np.linalg.eig(laplacian_matrix)

        # Keep the n_clusters smaller eigenvalues
        sort_ind = np.argsort(eigval)[: self.n_vectors]

        # Sort and plot eigenvalues
        #eigval = np.sort(eigval)

        # Initialize new array for the transormed data
        transormed_data = np.zeros((len(laplacian_matrix), self.n_vectors-1), dtype=np.float64)

        # Create transformed data
        for i in range(0, len(laplacian_matrix)):
            # Ignore first eigenvalue as it is close or equal to 0
            for j in range(1, self.n_vectors):
                transormed_data[i][j-1] = eigvec[i, np.asscalar(sort_ind[j])]
        return transormed_data


    """
    Transform and return data to 2D using LocallyLinearEmbedding
    """
    def LLE(self, data):
        if self.lle is None:
            self.lle = LocallyLinearEmbedding(n_components=2)
            self.lle.fit(data)

        return self.lle.transform(data)


    """
    Calculate and return the nearest neighbors graph which depicts the distances between each point to another
    The graph connects only the items with at most limit distance between them and everything else is zero resulting in a sparse matrix
    Default limit is 0.4
    """
    def NNGraph(self, data, limit=0.4):
        # Create the nearest neighbors graph
        graph = radius_neighbors_graph(data, limit, mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
        graph = graph.toarray()
        return graph


    """
    Calculate and return the similarity matrix using the rbf kernel
    """
    def SimilarityMatrix(self, data, limit=0.4):
        size = len(data)

        # Initialize array of size x size with zeros
        similarity_matrix = np.zeros((size, size), dtype=np.float64)
        for i in range(0, size):
            for j in range(0, size):
                if i != j:
                    value = self.rbf(data[i], data[j], 0.5)
                    #if value <= limit:
                        #similarity_matrix[i][j] = value
                    similarity_matrix[i][j] = value

        return similarity_matrix


    """
    Calculate and return the Laplacian matrix
    """
    def LaplacianMatrix(self, similarity_matrix):

        D = np.zeros(similarity_matrix.shape)
        w = np.sum(similarity_matrix, axis=0)
        D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
        return D.dot(similarity_matrix).dot(D)


    """
    Run sklearn's Spectral Cluster method for comparison
    """
    def SklearnSP(self, x_train, affinity='rbf'):
        model = cluster.SpectralClustering(n_clusters=self.n_clusters, affinity=affinity)
        model.fit(x_train)
        y_predict = model.fit_predict(x_train)
        return y_predict


    """
    Return exp(−||a − b||^2/s^2) where s = sigma
    """
    def rbf(self, a, b, sigma):

        result = math.exp( -math.pow( self.VectorLength( self.VectorSub(a, b) ) , 2) / math.pow(sigma, 2) )
        return result


    """
    Return the legth of vector v
    """
    def VectorLength(self, v):
        sum = 0
        for item in v:
            sum += item * item
        return math.sqrt(sum)


    """
    Return the result of the subtraction a - b where a and b are vectors of the
    same length
    """
    def VectorSub(self, a, b):
        if (len(a) != len(b)):
            return None

        v = np.zeros(len(a), dtype=np.float64)
        for i in range(0, len(a)):
            v[i] = a[i] - b[i]
        return v


    """
    Visualize 2D data
    """
    def visualize2D(self, x, y, c=None, title='', filename=None):
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.set_title(title, fontsize=16)
        cmap = 'viridis'
        dot_size=50
        # Check if there are different colored items in the plot
        if c is not None:
            for i in range(0, self.n_clusters-1) :
                temp_c = c[ (i*self.size) : (i+1) * self.size]
                ax.scatter(x, y, c=temp_c, s=dot_size, cmap=cmap)
        else:
            ax.scatter(x, y, s=dot_size)
        # Save to file or display plot
        if filename is not None:
            plt.savefig(filename + '.png')
            plt.clf()
            plt.close()
        else:
            plt.show()


    """
    Log results
    """
    def logResults(self, y_test, prediction, sklearn=False, affinity='rbf', laplacian='custom'):
        if sklearn is True:
            algorithm = 'SKLearn Spectral Clustering'
        else:
            algorithm = 'Custom Spectral Clustering'
        # Calculate precision, recall, f1
        result = metrics.precision_recall_fscore_support(y_test, prediction, average='macro')
        self.results = self.results.append({ 'Algorithm': algorithm, 'Affinity': affinity,
                          'N_Vectors': str(self.n_vectors),
                          'Laplacian': laplacian, 'Precision':  float("%0.3f"%result[0]),
                          'Recall': float("%0.3f"%result[1]), 'F1': float("%0.3f"%result[2])}, ignore_index=True)


    """
    Setup results dataframe object
    """
    def setupResults(self):
        self.results = pd.DataFrame(columns=['Algorithm', 'Affinity', 'Laplacian', 'N_Vectors', 'Precision', 'Recall', 'F1'])
