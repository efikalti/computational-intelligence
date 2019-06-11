import logging
import math
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import metrics
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster
from sklearn.preprocessing import normalize
import numpy as np


import matplotlib.pyplot as plt
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
        self.affinity = ['rbf', 'nearest_neighbors']


    """
    Run Locally Linear Embedding and Spectral Clustering on the provided data
    LLE reduces the data to 2D
    Spectral Clustering runs for n_clusters, default is 2
    """
    def train(self, x_train, y_train, x_test, y_test, n_clusters=2):

        # Set number of clusters
        self.n_clusters = n_clusters
        # Set the size to the training set size
        self.size = len(x_train)
        # Create list with numbers from 1 to number of training items
        self.iterations = np.zeros(self.size)
        for i in range(0, self.size):
            self.iterations[i] = i+1

        # Apply Locally Linear Embedding on training and testing data
        x_train = self.LLE(x_train)
        x_test = self.LLE(x_test)

        # Plot training data
        self.visualize2D(x_train[:, 0], x_train[:, 1], c=y_train, title='Training data')

        self.SpectralClustering(x_train, y_train)


    """
    Run Spectral Clustering for these data with these parameters
    affinity=['rbf', 'nearest_neighbors'],
    Default is rbf kernel for similarity matrix,
    """
    def SpectralClustering(self, x_train, y_train, affinity='nearest_neighbors'):

        # Get similarity matrix for train data
        if affinity == 'nearest_neighbors':
            similarity_matrix = self.NNGraph(x_train)
        else:
            similarity_matrix = self.SimilarityMatrix(x_train)

        # Get degree matrix from similarity matrix
        degree_matrix = self.DegreeMatrix(similarity_matrix)

        # Get laplacian matrix from similarity matrix and degree matrix
        #laplacian_matrix = self.LaplacianMatrix(similarity_matrix=similarity_matrix, degree_matrix=degree_matrix)
        laplacian_matrix = csgraph.laplacian(similarity_matrix, normed=True)

        y_spec = self.transformDataToLaplacian(laplacian_matrix)

        model = cluster.KMeans(n_clusters=self.n_clusters, precompute_distances='auto', random_state=0)
        predicted = model.fit(y_spec).labels_

        print(predicted)
        self.visualize2D(x_train[:, 0], x_train[:, 1], c=predicted, title='Custom SpectralClustering')

        for i in range(0, len(y_train)):
            if y_train[i] == -1:
                y_train[i] = 0

        print(metrics.precision_recall_fscore_support(y_train, predicted, average='macro'))

        # Run with sklearns Spectral Clustering
        #self.SklearnSP(x_train)


    """
    Create the new data using the laplacian matrix and its eigenvalues and eigenvectors
    """
    def transformDataToLaplacian(self, laplacian_matrix):
        # Get eigenvalues and eigenvectors from the laplacian matrix
        eigval, eigvec = np.linalg.eig(laplacian_matrix)

        n_clusters = 5

        # Keep the n_clusters smaller eigenvalues
        sort_ind = np.argsort(eigval)[: n_clusters]

        # Sort and plot eigenvalues
        eigval = np.sort(eigval)
        self.visualize2D(self.iterations, eigval)

        # Initialize new array for the transormed data
        transormed_data = np.zeros((len(laplacian_matrix), n_clusters-1), dtype=np.float64)

        # Create transformed data
        for i in range(0, len(laplacian_matrix)):
            # Ignore first eigenvalue as it is close or equal to 0
            for j in range(1, n_clusters):
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
        # A = kneighbors_graph(X_mn, 2, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False)
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
    Calculate and return the Degree matrix
    """
    def DegreeMatrix(self, similarity_matrix):
        size = len(similarity_matrix)

        # Initialize array of size x size with zeros
        degree_matrix = np.zeros((size, size), dtype=np.float64)

        # Calculate sum of every row and set it in the diagonal
        index = 0
        for row in similarity_matrix:
            sum = 0
            for item in row:
                sum += item
            degree_matrix[index][index] = sum
            index += 1

        return degree_matrix


    """
    Calculate and return the Laplacian matrix
    """
    def LaplacianMatrix(self, similarity_matrix, degree_matrix):
        #return degree_matrix - similarity_matrix
        D = np.zeros(similarity_matrix.shape)
        w = np.sum(similarity_matrix, axis=0)
        D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
        return D.dot(similarity_matrix).dot(D)


    """
    Run sklearn's Spectral Cluster method for comparison
    """
    def SklearnSP(self, x_train):
        model = cluster.SpectralClustering(n_clusters=self.n_clusters, affinity='rbf')
        model.fit(x_train)
        y_predict = model.fit_predict(x_train)
        self.visualize(x_train, y_predict, title='SKLearn SpectralClustering')


    """
    Return exp(−||a − b||^2/s^2) where s = sigma
    """
    def rbf(self, a, b, sigma):
        #delta = np.array(abs(np.subtract(a, b)))
        #distance = (np.square(delta).sum())
        #c = np.exp(-(distance**2)/(sigma**2))
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
        ax.set_title(title, fontsize=18)
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
            pyplot.savefig(filename + '.png')
            pyplot.clf()
        else:
            plt.show()
