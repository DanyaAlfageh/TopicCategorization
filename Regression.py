import numpy as np
import pandas as pd
import scipy.sparse as ss
import datetime
from Confusion import ConfusionMatrix
from IO import DataOut
import os

"""
The Linear Regression class creates the conditional likelihood estimate and performs gradient descent
to find the weights that best classify new examples. It can also perform classification on a validation
or testing set given a set of weights stored in a scipy csr matrix in .npz file format
"""
class LinearRegression():

    def __init__(self, trainingData = [], testingData =[], predictionData=[]):
        self.trainingData = trainingData
        self.testingData = testingData
        self.predictionData = predictionData
        #loading the csr representation of the training set
        loader = np.load('data/dense/denseRepTraining.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        self.trainingMatrix = ss.csr_matrix(args, shape=loader['shape'])
        #Loading the csr representation of the validation set
        loaderVal = np.load('data/dense/denseRepValidation.npz')
        argsVal = (loaderVal['data'], loaderVal['indices'], loaderVal['indptr'])
        self.validationMatrix = ss.csr_matrix(argsVal, shape=loaderVal['shape'])
        #Loading the csr representation of the testing set
        loaderTesting = np.load('data/dense/denseRepTesting.npz')
        argsTesting = (loaderTesting['data'], loaderTesting['indices'], loaderTesting['indptr'])
        self.testingMatrix = ss.csr_matrix(argsTesting, shape=loaderTesting['shape'])
        #setting up the X matrix for the calculations
        self.x = self.make_x_matrix()
        shape = self.x.shape
        #The number of training examples
        self.m = shape[0]
        #The number of attributes each example has
        self.n = shape[1]-1
        #The delta matrix which indicates which class every example is in
        self.deltaMatrix = self.make_delta_matrix()
        #The weight matrix
        self.w = self.make_weight_matrix()
        #Creating the conditional likelihood matrix
        self.make_prob_matrix()
        #Normalizing the validation set in order to make predications on it
        self.validationNormalized = self.normalize_matrix_for_classification(useValidation = True)
        #Normalizing the testing set in order to make predications on it
        self.testingNormalized = self.normalize_matrix_for_classification(useValidation = False)

    """
    The make_delta_matrix function finds the classification for all the training examples
    and puts them into a k x m matrix where for each example(column) there is a one in the row
    corresponding to its class and 0 in every other row.
    """
    def make_delta_matrix(self):
        print('Inside linear Reg')
        dataDict = dict()
        dataToMatrix = dict()
        Yclassifications = list()
        for i in range(20):
            dataDict[i] = []
            dataToMatrix[i] = []
        #Getting all of the classifications
        for j in range(self.m):
            row = self.trainingMatrix.data[self.trainingMatrix.indptr[j]:self.trainingMatrix.indptr[j+1]]
            classification = row[-1]
            Yclassifications.append(classification)
            dataDict[classification-1].append(j)
        #Creating a dictionary that represents the delta matrix
        for k in range(20):
            for p in range(self.m):
                if p in dataDict[k]:
                    dataToMatrix[k].append(1)
                else:
                    dataToMatrix[k].append(0)
        #Converting the dictionary representation into a scipy sparse representation.
        dataFrame = pd.DataFrame(dataToMatrix)
        dfTranspose = dataFrame.transpose()
        sparseCoo = ss.coo_matrix(dfTranspose)
        csr = sparseCoo.tocsr()
        #Delta matrix
        return csr

    """
    Creating and normalizing the X matrix. The colums are normalized by the max value in each
    column thus mapping all values between 0 and 1.
    """
    def make_x_matrix(self):
        colList = list(range(61188))
        #Creating column of all ones
        ones = np.ones(9600, dtype=np.float64)
        #Adding th column of all ones to first coloumn of matrix
        x = ss.hstack((ones[:,None], self.trainingMatrix[:, colList]))
        xCols = x.tocsc()
        #Getting maxes of all columns
        self.maxes = xCols.max(axis=0).toarray()
        #Normalizing each column
        xColsNormalized = xCols/self.maxes[:, 1]
        #Converting back to csr
        xNorm = ss.csr_matrix(xColsNormalized)
        return xNorm

    """
    Creating the weight matrix with inital values randomly chosen between 0 and 1
    """
    def make_weight_matrix(self):
        randWeights = np.random.random_sample((20, self.n+1))
        weightFrame = pd.DataFrame(randWeights)
        sparseWeights = ss.coo_matrix(weightFrame)
        return sparseWeights.tocsr()

    """
    Creating the probablity matrix. This function is used for creating and updating the
    conditional likelihood for the gradient descent algorithm. As well creating the probablity
    matrix used to classify either the validation or the testing sets.
    """
    def make_prob_matrix(self, useForGD = True, useValidation = True, weights = None):
        #Use for gradient descent
        if useForGD:
            product = self.w.dot((self.x.transpose()))
            numOnes = self.m
        else:
            #Use for validation set
            if useValidation:
                product = weights * self.validationNormalized.transpose()
                numOnes = self.validationNormalized.shape[0]
            #Use for testing set
            else:
                product = weights * self.testingNormalized.transpose()
                numOnes = self.testingNormalized.shape[0]
        #taking the exponential of the the product
        expProduct = np.expm1(product)
        #Converting to csr matrix
        replace = ss.csr_matrix(expProduct)
        #Creating a row of all 1's
        ones = np.ones(numOnes, dtype=np.float64)
        rowsToKeep = list(range(19))
        #Replacing last row with vector of all 1's and converting to csr
        addOnes = ss.vstack((replace[rowsToKeep, :], ones[None, :]))
        modifiedCsr = ss.csr_matrix(addOnes)
        expCols = ss.csc_matrix(expProduct)
        #Normalizing the columns by the sum of each column, so that each column sums to 1
        sums = expCols.sum(axis=0)
        normalized = expCols/sums
        csr = ss.csr_matrix(normalized)

        return csr


    """
    Normalizes the validation and training sets' columns by the max  of the corresponding
    column int eh X matrix.
    """
    def normalize_matrix_for_classification(self, useValidation = True):
        colList = list(range(61188))
        #Normalizing validation set
        if useValidation:
            yC = ss.csc_matrix(self.validationMatrix)
            #Getting and storing classifications for validation set
            self.yClasses = yC[:, -1].toarray()
            #Adding column of 1's to validation set
            ones = np.ones(yC.shape[0], dtype=np.float64)
            y = ss.hstack((ones[:,None], self.validationMatrix[:, colList] ))
        else:
            yC = ss.csc_matrix(self.testingMatrix)
            #Adding column of 1's to training set
            ones = np.ones(yC.shape[0], dtype=np.float64)
            y= ss.hstack((ones[:,None], self.testingMatrix ))
        #Normalizing the columns by the maxes of X columns
        yCols = y.tocsc()
        yColsNormalized = yCols/self.maxes[:, 1]
        return ss.csr_matrix(yColsNormalized)

    """
    The gradient_descent function computes the weights used for classification for Logistic Regression,
    and then writes the computed weights to file. The arguemts to the function are the parameters that
    can be used to tune the gradien descent, learning rate, penalty term, and number of max iterations
    """
    def gradient_descent(self, learningRate, penaltyTerm, maxIterations):
        print('Starting GD with ', maxIterations, 'iterations. At: ', str(datetime.datetime.now()))
        iters = 0
        val0 = 0
        #Iterating maxIterations number of times
        while iters <= maxIterations:
            if (iters % 500) == 0:
                print('On step', iters,  'At:', str(datetime.datetime.now()))
                val1 = self.classifyData()
                if val1 > val0:
                    val0 = val1
                    weights = self.w
            #Applyting the calculations to get next set of weights
            probMat = self.make_prob_matrix()
            diff = self.deltaMatrix - probMat
            wt = self.w + learningRate*((diff.dot(self.x) - penaltyTerm*self.w))
            self.w = ss.csr_matrix(wt)
            iters += 1
        #Getting the time it took to complete the gradient descent algorithm
        now = datetime.datetime.now()
        """
        attributes = {
            'data': self.w.data,
            'indices': self.w.indices,
            'indptr': self.w.indptr,
            'shape': self.w.shape
        }
        """
        attributes = {
            'data': weights.data,
            'indices': weights.indices,
            'indptr': weights.indptr,
            'shape': weights.shape
        }
        #Writing calculated weights to a file, so they can be used later
        np.savez('weights/weights'+'LR'+str(learningRate)+'PT' +str(penaltyTerm)+'iters'+str(iters)+'.npz', **attributes)
        print('Finsihed GD with ', iters, 'iterations. At: ',str(datetime.datetime.now()))

    """
    Given either the validation or testing set, this function will classify the examples given, either
    with the current weights being calculated in gradient descent or loading weights from file, to compare
    or to submitt to the competition.
    """
    def classifyData(self, fileName = None, validation = True, createConfusion = False):
        weights = None
        #Loading weights from file
        if fileName !=None:
            #load weights here
            loader = np.load('weights/' + fileName)
            args = (loader['data'], loader['indices'], loader['indptr'])
            weights = ss.csr_matrix(args, shape=loader['shape'])
        #Using current weights. This is used in gradient descent to check progress of the algorithm.
        else:
            weights = self.w
        #Using the validation set
        if validation:
            matrix = self.validationNormalized
            #Applying the given weights
            classifer = self.make_prob_matrix(useForGD = False, useValidation = True, weights = weights)
            #Getting the class with the highest probability for each example
            classiferArgs = classifer.argmax(axis = 0)
            #Adding 1 as classes go from 1 to 20 and argmax will return indexes 0 to 19
            classifications = ss.csc_matrix(classiferArgs + 1)
            classification = classifications.toarray()
            #Lists to be used in making a confusion matrix
            pred = []
            act = []
            #Checking to see if deminsions match
            if classification.shape[1] != len(self.yClasses):
                print('len of classification', len(classification), 'len of yClasses', len(self.yClasses))
            #Have matching deminsions
            else:
                correctlyClassified = 0
                for i in range(len(self.yClasses)):
                    pred.append(int(classification[0][i]))
                    act.append(int(self.yClasses[i][0]))
                    if classification[0][i] == self.yClasses[i][0]:
                        correctlyClassified += 1
                percentCorrect = correctlyClassified/len(self.yClasses)
                print('percentCorrect:', percentCorrect)

            if createConfusion:
                confusion = ConfusionMatrix(pred, act)
            return percentCorrect
        #Using testing set
        else:
            matrix = self.testingNormalized
            #Getting the weights
            classifer = self.make_prob_matrix(useForGD =False, useValidation =False, weights =weights)
            classiferArgs = classifer.argmax(axis = 0)
            classifications = ss.csc_matrix(classiferArgs + 1)
            classification = classifications.toarray()
            #Writing out to file so it can be submitted for the competition
            pred = DataOut()
            for i in range(classification.shape[1]):
                pred.add(i+12001, int(classification[0][i]))
            pred.write()
    """
    Method created for iterating through weight files and writing accuracies on the validation set
    to file
    """
    def find_score_for_all_computed_weights(self):
        calculatedWeights = os.listdir('weights')
        toFile = DataOut('data/LinRegAccuracies.csv')
        for weight in calculatedWeights:
            percentCorrect = self.classifyData(fileName=weight)
            toFile.add(weight, percentCorrect)
        toFile.write()
