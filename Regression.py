import numpy as np
import pandas as pd
import scipy.sparse as ss
import datetime
from Confusion import ConfusionMatrix
from IO import DataOut

class LinearRegression():

    def __init__(self, trainingData = [], testingData =[], predictionData=[]):
        self.trainingData = trainingData
        self.testingData = testingData
        self.predictionData = predictionData
        loader = np.load('data/dense/denseRepTraining.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        self.trainingMatrix = ss.csr_matrix(args, shape=loader['shape'])

        loaderVal = np.load('data/dense/denseRepValidation.npz')
        argsVal = (loaderVal['data'], loaderVal['indices'], loaderVal['indptr'])
        self.validationMatrix = ss.csr_matrix(argsVal, shape=loaderVal['shape'])

        loaderTesting = np.load('data/dense/denseRepTesting.npz')
        argsTesting = (loaderTesting['data'], loaderTesting['indices'], loaderTesting['indptr'])
        self.testingMatrix = ss.csr_matrix(argsTesting, shape=loaderTesting['shape'])

        self.normValues = dict()
        self.x = self.make_x_matrix()
        shape = self.x.shape
        self.m = shape[0]
        self.n = shape[1]-1
        self.deltaMatrix = self.make_delta_matrix_and_Y_vector()
        self.w = self.make_weight_matrix()

        self.make_prob_matrix()
        self.validationNormalized = self.normalize_matrix_for_classification(useValidation = True)
        self.testingNormalized = self.normalize_matrix_for_classification(useValidation = False)

    def make_delta_matrix_and_Y_vector(self):
        print('Inside linear Reg')
        dataDict = dict()
        dataToMatrix = dict()
        Yclassifications = list()
        for i in range(20):
            dataDict[i] = []
            dataToMatrix[i] = []
        for j in range(self.m):
            row = self.trainingMatrix.data[self.trainingMatrix.indptr[j]:self.trainingMatrix.indptr[j+1]]
            classification = row[-1]
            Yclassifications.append(classification)
            dataDict[classification-1].append(j)

        self.Y = Yclassifications

        for k in range(20):
            for p in range(self.m):
                if p in dataDict[k]:
                    dataToMatrix[k].append(1)
                else:
                    dataToMatrix[k].append(0)
        dataFrame = pd.DataFrame(dataToMatrix)
        dfTranspose = dataFrame.transpose()
        sparseCoo = ss.coo_matrix(dfTranspose)
        csr = sparseCoo.tocsr()
        print(csr.shape)
        print(sum(csr.sum(axis=1)))

        #Delta matrix
        return csr

    def make_x_matrix(self):
        #X matrix with normalization
        print(str(datetime.datetime.now()))
        colList = list(range(61188))
        ones = np.ones(9600, dtype=np.float32)
        x = ss.hstack((ones[:,None], self.trainingMatrix[:, colList]))
        xCols = x.tocsc()
        self.maxes = xCols.max(axis=0).toarray()
        xColsNormalized = xCols/self.maxes[:, 1]
        xNorm = ss.csr_matrix(xColsNormalized)
        return xNorm


    def make_weight_matrix(self):
        randWeights = np.random.random_sample((20, self.n+1))
        weightFrame = pd.DataFrame(randWeights)
        sparseWeights = ss.coo_matrix(weightFrame)
        return sparseWeights.tocsr()


    def make_prob_matrix(self, useForGD = True, useValidation = True, weights = None):
        #print('w is: ', self.w.shape)
        if useForGD:
            product = self.w.dot((self.x.transpose()))
            numOnes = self.m
        else:
            if useValidation:
                product = weights * self.validationNormalized.transpose()
                numOnes = self.validationNormalized.shape[0]
            else:
                product = weights * self.testingNormalized.transpose()
                numOnes = self.testingNormalized.shape[0]
        expProduct = np.expm1(product)

        replace = ss.csr_matrix(expProduct)
        ones = np.ones(numOnes, dtype=np.float32)
        rowsToKeep = list(range(19))
        addOnes = ss.vstack((replace[rowsToKeep, :], ones[None, :]))
        modifiedCsr = ss.csr_matrix(addOnes)

        expCols = ss.csc_matrix(expProduct)
        sums = expCols.sum(axis=0)
        normalized = expCols/sums
        csr = ss.csr_matrix(normalized)

        #print('expProduct is: ', expProduct.shape)
        return csr



    def normalize_matrix_for_classification(self, useValidation = True):
        #Need to just create and load testing matrix
        #Need to remove and store classifications ie last column
        colList = list(range(61188))
        if useValidation:
            yC = ss.csc_matrix(self.validationMatrix)
            self.yClasses = yC[:, -1].toarray()
            ones = np.ones(yC.shape[0], dtype=np.float32)
            y = ss.hstack((ones[:,None], self.validationMatrix[:, colList] ))
        else:
            yC = ss.csc_matrix(self.testingMatrix)
            ones = np.ones(yC.shape[0], dtype=np.float32)
            y= ss.hstack((ones[:,None], self.testingMatrix ))

        yCols = y.tocsc()
        yColsNormalized = yCols/self.maxes[:, 1]
        return ss.csr_matrix(yColsNormalized)

    def gradient_descent(self, learningRate, penaltyTerm, maxIterations):
        print('Starting GD with ', maxIterations, 'iterations. At: ', str(datetime.datetime.now()))
        iters = 0

        while iters <= maxIterations:
            if (iters % 1000) == 0:
                print('On step', iters,  'At:', str(datetime.datetime.now()))
                self.classifyData()

            probMat = self.make_prob_matrix()
            wt = self.w + learningRate*(((self.deltaMatrix - probMat) *self.x) - penaltyTerm*self.w)
            self.w = ss.csr_matrix(wt)
            iters += 1

        now = datetime.datetime.now()
        timeForFileName = str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)
        attributes = {
            'data': self.w.data,
            'indices': self.w.indices,
            'indptr': self.w.indptr,
            'shape': self.w.shape
        }
        np.savez('weights/weights'+'LR'+str(learningRate)+'PT' +str(penaltyTerm)+'iters'+str(maxIterations)+'.npz', **attributes)
        print('Finsihed GD with ', maxIterations, 'iterations. At: ',str(datetime.datetime.now()))

    def classifyData(self, fileName = None, validation = True, createConfusion = False):
        weights = None
        if fileName !=None:
            #load weights here
            loader = np.load('weights/' + fileName + '.npz')
            args = (loader['data'], loader['indices'], loader['indptr'])
            weights = ss.csr_matrix(args, shape=loader['shape'])
        else:
            weights = self.w

        if validation:
            matrix = self.validationNormalized
            classifer = self.make_prob_matrix(useForGD = False, useValidation = True, weights = weights)
            #print('shape of classifer', classifer.shape)
            classiferArgs = classifer.argmax(axis = 0)
            classifications = ss.csc_matrix(classiferArgs + 1)
            classification = classifications.toarray()
            pred = []
            act = []
            if classification.shape[1] != len(self.yClasses):
                print('len of classification', len(classification), 'len of yClasses', len(self.yClasses))
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
        else:
            matrix = self.testingNormalized
            classifer = self.make_prob_matrix(useForGD =False, useValidation =False, weights =weights)
            classiferArgs = classifer.argmax(axis = 0)
            classifications = ss.csc_matrix(classiferArgs + 1)
            classification = classifications.toarray()
            pred = DataOut()
            for i in range(classification.shape[1]):
                pred.add(i+12001, int(classification[0][i]))
            pred.write()
