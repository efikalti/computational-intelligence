from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot

class Decomposition:

    """
    Constructor
    """
    def __init__(self):
        self.lda = None
        self.kpca = None

        self.kernel = ['linear', 'rbf']
        self.Gamma = [0.01, 0.1, 1]
        self.components = [15, 10, 8, 1]


    """
    Create transform to reduce data dimensionality
    """
    def fit(self, x, y=None, kernel='rbf', gamma=1, n_components=10):
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
        X_transformed = self.kpca.fit_transform(X=x)
        self.lda = LinearDiscriminantAnalysis(n_components=2)
        X_transformed = self.lda.fit_transform(X=X_transformed, y=y)
        return X_transformed


    """
    Apply transform to reduce data dimensionality
    """
    def transform(self, x, y=None):
        if self.kpca is None:
            return x
        else:
            X_transformed = self.kpca.transform(X=x)
            X_transformed = self.lda.transform(X=X_transformed)
        return X_transformed


    """
    Create visualization for data reduced to 2 dimensions
    testing multiple variables
    """
    def test_visualize(self, x, y):
        for kernel in self.kernel:
            if kernel == 'rbf' or kernel == 'sigmoid':
                for gamma in self.Gamma:
                    self.lda = LinearDiscriminantAnalysis()
                    X_transformed = self.lda.fit_transform(X=x, y=y)
                    self.kpca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma)
                    X_transformed = self.kpca.fit_transform(X=X_transformed, y=y)
                    filename = 'lda-kpca_' + kernel + '_' + str(gamma)
                    self.plot(X_transformed, y, 'Dimensionality reduction LDA - KPCA with ' + kernel + ' kernel, Gamma=' + str(gamma), filename)
            else:
                self.lda = LinearDiscriminantAnalysis()
                X_transformed = self.lda.fit_transform(X=x, y=y)
                self.kpca = KernelPCA(n_components=2, kernel=kernel)
                X_transformed = self.kpca.fit_transform(X=X_transformed, y=y)
                filename = 'lda-kpca_' + kernel
                self.plot(X_transformed, y,'Dimensionality reduction LDA - KPCA with ' + kernel + ' kernel', filename)
        for component in self.components:
            for kernel in self.kernel:
                if kernel == 'rbf' or kernel == 'sigmoid':
                    for gamma in self.Gamma:
                        self.kpca = KernelPCA(n_components=component, kernel=kernel, gamma=gamma)
                        X_transformed = self.kpca.fit_transform(X=x, y=y)
                        self.lda = LinearDiscriminantAnalysis()
                        X_transformed = self.lda.fit_transform(X=X_transformed, y=y)
                        filename = 'kpca-lda_' + kernel + '_' + str(gamma) + '_' + str(component)
                        self.plot(X_transformed, y, 'Dimensionality reduction KPCA - LDA with ' + kernel + ' kernel, Gamma=' + str(gamma) + ', component=' + str(component), filename)
                else:
                    self.kpca = KernelPCA(n_components=component, kernel=kernel)
                    X_transformed = self.kpca.fit_transform(X=x, y=y)
                    self.lda = LinearDiscriminantAnalysis()
                    X_transformed = self.lda.fit_transform(X=X_transformed, y=y)
                    filename = 'kpca-lda_' + kernel + '_' + str(component)
                    self.plot(X_transformed, y,'Dimensionality reduction KPCA - LDA with ' + kernel + ' kernel, components=' + str(component), filename)


    """
    Visualize a single kpca-lda using the transform stored in the object
    """
    def visualize(self, x, y, kernel='rbf', gamma=0.1, component=10, fileprefix=''):
        filename = fileprefix + 'kpca-lda_' + kernel + '_' + str(gamma) + '_' + str(component)
        self.plot(x, y, 'Dimensionality reduction KPCA - LDA with ' + kernel + ' kernel, Gamma=' + str(gamma) + ', component=' + str(component), filename)



    """
    Plot data
    """
    def plot(self, data, labels, title, filename):
        x = {1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        y = {1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        for i in range(0, len(data)):
            label = labels[i]
            x[label].append(data[i, 0])
            y[label].append(data[i, 1])

        pyplot.figure(figsize=[13, 6])
        pyplot.plot( x[1],  y[1], 'bo', label='Bronze League')
        pyplot.plot( x[2],  y[2], 'ro', label='Silver League')
        pyplot.plot( x[3],  y[3], 'yo', label='Gold League')
        pyplot.plot( x[4],  y[4], 'co', label='Platinum League')
        pyplot.plot( x[5],  y[5], 'mo', label='Diamond League')
        pyplot.plot( x[6],  y[6], 'go', label='Master League')
        pyplot.plot( x[7],  y[7], 'ko', label='GrandMaster League')
        pyplot.plot( x[8],  y[8], 'bs', label='Professional League')
        pyplot.legend()
        pyplot.title(label=title)
        pyplot.savefig('logs/kpca-lda/' + filename + '.png')
        pyplot.clf()
        pyplot.close()
