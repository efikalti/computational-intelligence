import numpy as np
import pandas as pd
import logging
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import random

class DataLoader:

    """
    Constructor
    Initializes the class variables necessary for preprocessing the data
    """
    def __init__(self):
        # Values for normalizing
        self.numOfFeatures = 0
        self.rows = 0
        self.columns = 0
        self.Y = None
        self.X = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.multi_y_train = None
        self.multi_y_test = None
        self.scalers = None

    """
    Read function
    - Opens and reads the data from the given file using the pandas library to
      autoformat the rows and columns
    - Works for csv files
    - numOfFeatures is required to assign the number of features
    - Function by defaults assigns the class index to be the first column
    """
    def read(self, filename, numOfFeatures, classIndex=0, dtype=np.float64):
        # Read datas from csv file
        data = pd.read_csv(filename, dtype=dtype, na_values=['?'])

        # Get number or rows and columns
        self.rows = data.shape[0]
        self.cols = data.shape[1]

        # Get number of features
        self.numOfFeatures = numOfFeatures
        self.scalers = []

        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)

        # Separate data variables
        self.X = data.iloc[0:self.rows , classIndex + 1:self.cols]
        self.removeFalseData()
        self.X = self.X.fillna(self.X.mean())
        self.X = self.X.values

        # Separate class variable
        self.Y = data.iloc[0:self.rows, classIndex].values

        # Separate train and test data from original dataset to 60% and 40%
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, random_state=0, train_size=0.6)
        self.multi_y_train = np.array(self.y_train, copy=True, dtype=np.int32)
        self.multi_y_test = np.array(self.y_test, copy=True, dtype=np.int32)
        self.y_train = np.array(self.y_train, copy=True, dtype=np.int32)
        self.y_test = np.array(self.y_test, copy=True, dtype=np.int32)

        self.calculateScalers()

        # Clear data
        data = None


    """
    Creates binary labels according to separator
    """
    def scaleToBinary(self, separator):
        # Assign 1 or -1 according to condition
        for i in range(0, len(self.y_train)):
            if self.y_train[i] < separator:
                self.y_train[i] = 1
            else:
                self.y_train[i] = -1
        for i in range(0, len(self.y_test)):
            if self.y_test[i] < separator:
                self.y_test[i] = 1
            else:
                self.y_test[i] = -1


    """
    Scale train and test data values between 0.0 and 1.0
    """
    def normalize(self):
        # Scale all features of X
        for feature in range(0, self.numOfFeatures):
            self.x_train = self.scaleMinMax(feature, self.x_train)
            self.x_test = self.scaleMinMax(feature, self.x_test)


    """
    Count data for each label
    """
    def labelCount(self, len):
        count_l = np.zeros(len)
        for label in self.Y:
            count_l[int(label) - 1] += 1
        return count_l


    """
    Count data for each binary label
    """
    def binaryLabelCount(self, separator):
        count_l = np.zeros(2)
        for label in self.Y:
            if label >= separator:
                count_l[0] += 1
            else:
                count_l[1] += 1
        return count_l


    """
    Replace false data entry
    """
    def removeFalseData(self):
        # For TotalHours > 1000000
        max = 0
        index = 0
        for i in range(0, self.rows):
            if (self.X.at[i, 'TotalHours'] == 1000000):
                index = i
                self.X.at[index, 'TotalHours'] = self.X.mean()['TotalHours']
                return


    """
    Create plots for some of the data dimensions
    """
    def visualize(self):
        # Set data for plot
        age = {1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        hpw = {1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        th = {1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        apm = {1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        sbh = {1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
        for i in range(0, len(self.Y)):
            label = self.Y[i]
            age[label].append(self.X[i, 0])
            hpw[label].append(self.X[i, 1])
            th[label].append(self.X[i, 2])
            apm[label].append(self.X[i, 3])
            sbh[label].append(self.X[i, 4])

        # Plot data
        self.plot(age, hpw, 'Age', 'Hours Per Week', 'Age and Hours per week', 'age-hpw')
        self.plot(age, th, 'Age', 'Total Hours', 'Age and Total hours', 'age-th')
        self.plot(age, apm, 'Age', 'APM', 'Age and APM', 'age-apm')
        self.plot(age, sbh, 'Age', 'Select By Hotkeys', 'Age and Select by Hotkeys', 'age-sbh')
        self.plot(hpw, th, 'Hours Per Week', 'Total Hours', 'Hours Per Week and Total hours', 'hpw-th')
        self.plot(hpw, apm, 'Hours Per Week', 'APM', 'Hours Per Week and APM', 'hpw-apm')
        self.plot(hpw, sbh, 'Hours Per Week', 'Select By Hotkeys', 'Hours Per Week and Select By Hotkeys', 'hpw-sbh')
        self.plot(apm, sbh, 'APM', 'Select By Hotkeys', 'APM and Select By Hotkeys', 'apm-sbh')


    """
    Plot data
    """
    def plot(self, xarg, yarg, xlabel, ylabel, title, filename):
        pyplot.figure(figsize=[13, 6])
        pyplot.plot( xarg[1],  yarg[1], 'bo', label='Bronze League')
        pyplot.plot( xarg[2],  yarg[2], 'ro', label='Silver League')
        pyplot.plot( xarg[3],  yarg[3], 'yo', label='Gold League')
        pyplot.plot( xarg[4],  yarg[4], 'co', label='Platinum League')
        pyplot.plot( xarg[5],  yarg[5], 'mo', label='Diamond League')
        pyplot.plot( xarg[6],  yarg[6], 'go', label='Master League')
        pyplot.plot( xarg[7],  yarg[7], 'ko', label='GrandMaster League')
        pyplot.plot( xarg[8],  yarg[8], 'bs', label='Professional League')
        pyplot.legend()
        pyplot.title(label=title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)
        pyplot.savefig('logs/' + filename + '.png')
        pyplot.clf()


    """
    Scaler function
    - Uses the formula scaled = (value - min) / (max - min)
    - Range between [0 - 1]
    """
    def scaleMinMax(self, index, x):

        # Initialize to the first element of the column with this index
        min = self.scalers[index].get("min")
        max = self.scalers[index].get("max")

        # Cache max - min value
        max_min = max - min

        # Scale feature according to min and max findings
        for i in range(0, len(x)):
            rescaled_x = (x[i, index] - min) / max_min
            x[i, index] = rescaled_x
        return x


    """
    Scaler function
    - Uses the formula scaled = (value - min) / (max - min)
    - Range between [0 - 1]
    """
    def calculateScalers(self):
        for i in range(0, self.numOfFeatures):
            min = self.x_train[0, i]
            max = self.x_train[0, i]
            # Find min and max values for this feature
            for j in range(0, len(self.x_train)):
                element = self.x_train[j, i]
                if element < min:
                    min = element
                if element > max:
                    max = element
            # Save min max
            self.scalers.append({'min': min, 'max': max})
