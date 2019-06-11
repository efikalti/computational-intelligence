import logging
from dataLoader import DataLoader
from mnistReader import MNistReader
from clustering import Cluster
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
    starcraft_sp_test()

    # Run svm for mnist dataset
    #mnist_sp_test()


"""
Function that runs binary and multi-class Spectral Clustering models on the starcraft 2 dataset (SkillCraft1_Dataset.csv)
This function uses the dataLoader python class which is designed to load and process this specific dataset
"""
def starcraft_sp_test():

    # Create DataLoader instance to load and format data
    dataLoader = DataLoader()

    logging.info("Program started")

    logging.info("Loading starcraft data")
    # Read skillcraft dataset, the class index is the second column
    dataLoader.read(filename="data/SkillCraft1_Dataset.csv",  classIndex=1, numOfFeatures=15)

    # Normalize data values from 0 - 1
    #dataLoader.normalize()

    # Create new labels to fit into binary classification
    dataLoader.scaleToBinary(5)

    # Spectral Clustering

    # Binary
    clustering(dataLoader.x_train, dataLoader.y_train, writer_starcraft, 'starcraft-binary', multiple=True, binary=True)

    # Multiclass
    #clustering(dataLoader.x_train, dataLoader.multi_y_train, writer_starcraft, 'starcraft-multiclass', multiple=True, binary=False)

    # Write all the results
    writer_starcraft.save()


"""
Function that runs binary and multi-class Spectral Clustering models on the mnist dataset
This function uses the mnistReader python class which is designed to load and process this specific dataset
"""
def mnist_sp_test():

        # Create reader to load and save the training and test data
        reader = MNistReader()
        reader.load_training_data("./mnist-data/train-images.idx3-ubyte", "./mnist-data/train-labels.idx1-ubyte")

        # Spectral Clustering

        # Binary
        clustering(reader.X, reader.Y, writer_mnist, 'mnist-binary', multiple=False, binary=True)

        # Multiclass
        clustering(reader.X, reader.multiY, writer_mnist, 'mnist-multiclass', multiple=False, binary=False)

        # Write all the results
        writer_mnist.save()


"""
Run binary and multiclass Spectral Clustering
Save results in the provided writer
"""
def clustering(x_train, y_train, writer, sheet_name, multiple=False, binary=True):
    cluster = Cluster()
    cluster.setupResults()
    cluster.train(x_train, y_train, multiple=multiple, binary=binary)
    cluster.results.to_excel(writer, sheet_name=sheet_name)


if __name__ == "__main__":
    main()
