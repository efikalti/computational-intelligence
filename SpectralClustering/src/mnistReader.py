import os
import struct
from array import array
import logging

from PIL import Image
import numpy as np

###### Constants ######

# Number of bytes for integer data type in nmist
INTEGER_BYTE_SIZE = 4

##### IOStream class for input and output functions ######
class MNistReader:

    # Empty constructor
    def __init__(self):
        self.training_data = {}
        self.test_data = {}

        self.X = None
        self.Y = None
        self.multiY = None
        self.X_test = None
        self.Y_test = None
        self.multiYtest = None

        # Values for normalizing
        self.min = 0
        self.max = 255

    """
    Load the training data and their labels according to the nmist
    ubyte file format
    """
    def load_training_data(self, data_filename, labels_filename, normalize=True):
        if normalize is not True:
            self.max = 1

        logging.info("Started reading training data")
        # Read images in binary format
        datafile = open(data_filename, "rb")

        # Read the first four integers in the nmist training image file
        magic, size, rows, cols = struct.unpack(">IIII", datafile.read(4 * INTEGER_BYTE_SIZE))

        # Set loaded values
        self.training_data["magic"] = magic
        self.training_data["size"] = size
        self.training_data["pixels"] = rows

        # Read the first 100 pictures
        #size = 10000

        # Read each value and save it in image_data list
        logging.info("Reading train image data")
        image_data = array("B", datafile.read(size * 784))

        # Close training images file
        datafile.close()

        # Create images list to separate each image according to height and width pixels
        images = []

        # Number of pixels per image
        image_size = rows * cols

        # Fill images with zeros
        for i in range(size):
            images.append([0] * rows * cols)

        # Set each image from image_data
        # For example image 0, i.e. the first image of the dataset goes from 0 to 783
        # so for image[0] we set the values from [0 * 784 : 1 * 784)
        logging.info("Normalizing train image data...")
        for i in range(size):
            for j in range(image_size):
                # Set and normalize data with rescale
                images[i][j] = image_data[(i * image_size) + j] / self.max

        # Set class data
        self.X = images

        # Read the first two integers in the nmist training labels file
        datafile = open(labels_filename, "rb")
        magic_labels, size_labels = struct.unpack(">II", datafile.read(2 * INTEGER_BYTE_SIZE))

        #size_labels = 10000

        # Load the labels, change label value to odd or even according to the number value and set
        self.multiY = np.array(array("B", datafile.read(size_labels)))
        self.Y = self.convertLabelsToEvenAndOdd(np.copy(self.multiY))

        # Close training labels file
        datafile.close()
        logging.info("Finished reading train data.")


    """
    Load the testing data and their labels according to the nmist
    ubyte file format
    """
    def load_test_data(self, data_filename, labels_filename):
        logging.info("Started reading test data")
        # Read images in binary format
        datafile = open(data_filename, "rb")

        # Read the first four integers in the nmist test image file
        magic, size, rows, cols = struct.unpack(">IIII", datafile.read(4 * INTEGER_BYTE_SIZE))

        # Set loaded values
        self.test_data["magic"] = magic
        self.test_data["size"] = size
        self.test_data["pixels"] = rows

        size = 100

        # Read each value and save it in image_data list
        logging.info("Reading test image data...")
        image_data = array("B", datafile.read(size * 784))

        # Close training images file
        datafile.close()

        # Create images list to separate each image according to height and width pixels
        images = []

        # Number of pixels per image
        image_size = rows * cols

        # Fill images with zeros
        for i in range(size):
            images.append([0] * rows * cols)

        # Set each image from image_data
        # For example image 0, i.e. the first image of the dataset goes from 0 to 783
        # so for image[0] we set the values from [0 * 784 : 1 * 784)
        logging.info("Normalizing test image data...")
        for i in range(size):
            for j in range(image_size):
                # Set and normalize with rescale
                images[i][j] = image_data[(i * image_size) + j] / self.max

        # Set class data
        self.X_test = images

        # Read the first two integers in the nmist test labels file
        datafile = open(labels_filename, "rb")
        magic_labels, size_labels = struct.unpack(">II", datafile.read(2 * INTEGER_BYTE_SIZE))

        size_labels = 100

        # Load the labels, change label value to odd or even according to the number value and set
        self.multiYtest = np.array(array("B", datafile.read(size_labels)))
        self.Y_test = self.convertLabelsToEvenAndOdd(np.copy(self.multiYtest))

        # Close training labels file
        datafile.close()
        logging.info("Finished reading test data.")

    """
    Change labels for binary classification
    """
    def convertLabelsToEvenAndOdd(self, labels):
        new_labels = []
        for label in labels:
            if label % 2 == 0:
                # Assign 1 to every even number as label
                new_labels.append(1)
            else:
                # Assign 1 to every odd number as label
                new_labels.append(-1)
        return new_labels

    """
    Function to display an array as an image
    Only works for integer values from 0 - 255
    """
    def display(self, image_array, width, height):
        # Resize array to 2D array of width x height
        resized_array = np.resize(image_array, (width, height))

        # Create image and display
        im = Image.fromarray(resized_array)
        im.show()
