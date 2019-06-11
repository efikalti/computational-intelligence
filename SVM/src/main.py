import logging
from dataLoader import DataLoader
from mnistReader import MNistReader
from svm import SVM
from clustering import Cluster
from knClassifier import KNClassifier
import xlsxwriter
import pandas as pd

###### Constants ######
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
writer_starcraft = pd.ExcelWriter('logs/results_starcraft.xlsx', engine='xlsxwriter')
writer_mnist = pd.ExcelWriter('logs/results_mnist.xlsx', engine='xlsxwriter')

def main():
    # Setup logging to console
    logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

    # Run svm for starcraft dataset
    #starcraft_svm_test()

    # Run svm for mnist dataset
    mnist_svm_test()


"""
Function that runs binary and multi-class svm models on the starcraft 2 dataset (SkillCraft1_Dataset.csv)
This function uses the dataLoader python class which is designed to load and process this specific dataset
"""
def starcraft_svm_test():

    # Create DataLoader instance to load and format data
    dataLoader = DataLoader()

    logging.info("Program started")


    logging.info("Loading starcraft data")
    # Read skillcraft dataset, the class index is the second column
    dataLoader.read(filename="data/SkillCraft1_Dataset.csv",  classIndex=1, numOfFeatures=15)
    multi_label_count = dataLoader.labelCount(8)

    # Creates plots for a few of the data features
    # dataLoader.visualize()

    # Normalize data values from 0 - 1
    #dataLoader.normalize()

    # Create new labels to fit into binary classification
    dataLoader.scaleToBinary(5)
    label_count = dataLoader.binaryLabelCount(5)
    logging.info("Number of examples per class")
    logging.info("Casual - (1):           " + str(label_count[0]))
    logging.info("Hardcore - (-1):           " + str(label_count[1]))

    label_count = dataLoader.labelCount(8)
    logDataCount(label_count)

    """
    # Create SVM
    svm = SVM()

    # Train and predict for binary svm
    logging.info("Running SVM for binary classification")
    # Train for binary single run with these objects
    logging.info("Single binary SVM")
    svm.train(dataLoader.x_train, dataLoader.y_train, dataLoader.x_test, dataLoader.y_test)

    # Train and test binary svm multiple times for all available binary variables
    logging.info("Multiple runs with different parameters - binary SVM")
    svm.train(dataLoader.x_train, dataLoader.y_train, dataLoader.x_test, dataLoader.y_test, iterate=True)

    # Save binary results to excel sheet
    logging.info("Saving binary SVM results")
    svm.results.to_excel(writer_starcraft, sheet_name='binary-svm')


    # MULTI CLASS SVM
    logging.info("Running SVM for multiclass classification")


    # Train and predict for multi-class data using the linear svm from liblinear implementation
    logging.info("Running SVM for multiclass classification with liblinear implementation")
    svm.train(dataLoader.x_train, dataLoader.multi_y_train, dataLoader.x_test, dataLoader.multi_y_test, binary=False)
    logging.info("Saving multiclass liblinear results")
    svm.results.to_excel(writer_starcraft, sheet_name='multiclass-liblinear')

    # Train for multi-class single run with these objects using the libsvm implementation
    logging.info("Running SVM for multiclass classification with libsvm implementation")
    svm.train(dataLoader.x_train, dataLoader.multi_y_train, dataLoader.x_test, dataLoader.multi_y_test, binary=False, linear=False)
    logging.info("Saving multiclass libsvm results")
    svm.results.to_excel(writer_starcraft, sheet_name='multiclass-libsvm')

    # Train and test multi-class svm multiple times for all available multi-class variables
    logging.info("Running SVM for multiclass classification for all available multi-class variables")
    svm.train(dataLoader.x_train, dataLoader.multi_y_train, dataLoader.x_test, dataLoader.multi_y_test, iterate=True, binary=False)
    logging.info("Saving multiclass multiple-runs results")
    svm.results.to_excel(writer_starcraft, sheet_name='multiclass-multiple-variables')

    # Train and test multi-class svm multiple times with KPCA-LDA
    logging.info("Running SVM for multiclass classification with KPCA-LDA")
    svm.train(dataLoader.x_train, dataLoader.multi_y_train, dataLoader.x_test, dataLoader.multi_y_test, iterate=True, binary=False, decomposition=True)
    logging.info("Saving multiclass multiple-runs results")
    svm.results.to_excel(writer_starcraft, sheet_name='multiclass-kpca-lda')

    # KNN and NC
    nearest(dataLoader.x_train, dataLoader.y_train, dataLoader.x_test, dataLoader.y_test, dataLoader.multi_y_train, dataLoader.multi_y_test, writer_starcraft)
    """

    clustering(dataLoader.x_train, dataLoader.y_train, dataLoader.x_test, dataLoader.y_test)

    # Write all the results
    writer_starcraft.save()


"""
Function that runs binary and multi-class svm models on the mnist dataset
This function uses the mnistReader python class which is designed to load and process this specific dataset
"""
def mnist_svm_test():

        # Create reader to load and save the training and test data
        reader = MNistReader()
        reader.load_training_data("./mnist-data/train-images.idx3-ubyte", "./mnist-data/train-labels.idx1-ubyte")
        reader.load_test_data("./mnist-data/t10k-images.idx3-ubyte", "./mnist-data/t10k-labels.idx1-ubyte")

        # Create SVM object
        svm = SVM()
        svm.isMnist()
        """

        # Train and predict for binary svm
        logging.info("Running SVM for binary classification")
        #svm.train(reader.X, reader.Y, reader.X_test, reader.Y_test)
        logging.info("Saving binary results")
        #svm.results.to_excel(writer_mnist, sheet_name='binary')

        # MULTI CLASS SVM

        # Train and predict for binary svm
        logging.info("Multi Class SVM\n")

        # Train and predict for multi-class data using the linear svm from liblinear implementation

        # Train for multi-class single run with these objects liblinear implementation
        #svm.train(reader.X, reader.multiY, reader.X_test, reader.multiYtest, binary=False)
        logging.info("Saving multiclass liblinear results")
        #svm.results.to_excel(writer_mnist, sheet_name='multiclass-liblinear')

        # Train for multi-class single run with these objects using the libsvm implementation
        #svm.train(reader.X, reader.multiY, reader.X_test, reader.multiYtest, binary=False, linear=False)
        logging.info("Saving multiclass libsvm results")
        #svm.results.to_excel(writer_mnist, sheet_name='multiclass-libsvm')

        # Train for multi-class single run with these objects liblinear implementation KPCA-LDA
        svm.train(reader.X, reader.multiY, reader.X_test, reader.multiYtest, binary=False, decomposition=True, once=True)
        logging.info("Saving multiclass liblinear results with kpca-lda")
        svm.results.to_excel(writer_mnist, sheet_name='multiclass-liblinear-kpca-lda')

        # Train for multi-class single run with these objects using the libsvm implementation KPCA-LDA
        svm.train(reader.X, reader.multiY, reader.X_test, reader.multiYtest, binary=False, linear=False, decomposition=True, once=True, fileprefix='mnist_')
        logging.info("Saving multiclass libsvm results with kpca-lda")
        svm.results.to_excel(writer_mnist, sheet_name='multiclass-libsvm-kpca-lda')

        # KNN and NC
        nearest(reader.X, reader.Y, reader.X_test, reader.Y_test,  reader.multiY, reader.multiYtest, writer_mnist, once=True, starcraft=False)
        """

        clustering(reader.X, reader.Y, reader.X_test, reader.Y_test)


        # Write all the results
        writer_mnist.save()


"""
Run binary and multiclass Nearest Neighbor and Nearest NearestCentroid
Save results in the provided writer
"""
def nearest(x_train, y_train_binary, x_test, y_binary_test, y_multi_train, y_multi_test, writer, once=False, starcraft=True):
    # KNN and NC
    logging.info("Running K Nearest Neighbors and Nearest Centroid\n")
    knclassifier = KNClassifier()
    if starcraft is not True:
        knclassifier.isMnist()

    # Setup results object and run binary classification
    logging.info("Binary KNN & NC")
    #knclassifier.train(x_train, y_train_binary, x_test, y_binary_test, once=once)
    logging.info("Saving binary KNN & NC results")
    #knclassifier.results.to_excel(writer, sheet_name='binary-knn-nc')

    # Setup results object and run multiclass classification
    logging.info("Multiclass KNN & NC")
    #knclassifier.train(x_train, y_multi_train, x_test, y_multi_test, multiclass=True, once=once)
    logging.info("Saving multiclass KNN & NC results")
    #knclassifier.results.to_excel(writer, sheet_name='multiclass-knn-nc')

    # Run multiclass classification with KPCA-LDA
    logging.info("Multiclass KNN & NC with KPCA-LDA")
    knclassifier.train(x_train, y_multi_train, x_test, y_multi_test, multiclass=True, decomposition=True, once=once)
    logging.info("Saving binary KNN & NC results")
    knclassifier.results.to_excel(writer, sheet_name='multiclass-knn-nc-kpca-lda')


def clustering(x_train, y_train, x_test, y_test):
    cluster = Cluster()
    cluster.train(x_train, y_train, x_test, y_test)


"""
Log the data examples in each class
"""
def logDataCount(label_count):
    logging.info("Number of examples per class")
    logging.info("Bronze - 1:           " + str(label_count[0]))
    logging.info("Silver - 2:           " + str(label_count[1]))
    logging.info("Gold - 3:             " + str(label_count[2]))
    logging.info("Platinum - 4:         " + str(label_count[3]))
    logging.info("Diamond - 5:          " + str(label_count[4]))
    logging.info("Master - 6:           " + str(label_count[5]))
    logging.info("GrandMaster - 7:      " + str(label_count[6]))
    logging.info("Professional - 8:     " + str(label_count[7]))


if __name__ == "__main__":
    main()
