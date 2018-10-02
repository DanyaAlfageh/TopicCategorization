import numpy as np
import pandas as pd
import scipy.sparse as ss

class LinearRegression():

    def __init__(self, trainingData, testingData, predictionData,vocab):
        self.trainingData = trainingData
        self.testingData = testingData
        self.predictionData = predictionData


    def make_delta_matrix(self):
        loader = np.load('data/dense/denseRep.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])

        print('Inside linear Reg')
        row0 = matrix[0]
        row1 = matrix[1]
        dataDict = dict()
        dataToMatrix = dict()
        Yclassifications = list()
        for i in range(20):
            dataDict[i] = []
            dataToMatrix[i] = []
        for j in range(12000):
            row = matrix.data[matrix.indptr[j]:matrix.indptr[j+1]]
            classification = row[-1]
            Yclassifications.append(classification)
            dataDict[classification-1].append(j)

        for k in range(20):
            for p in range(12000):
                if p in dataDict[k]:
                    dataToMatrix[k].append(1)
                else:
                    dataToMatrix[k].append(0)
        dataFrame = pd.DataFrame(dataToMatrix)
        dfTranspose = dataFrame.transpose()
        sparseCoo = ss.coo_matrix(dfTranspose)
        #Delta matrix
        sparse = sparseCoo.tocsr()
        #print(sparse)

        #X matrix
        ones = np.ones(12000, dtype=np.uint32)
        X = ss.hstack((ones[:,None], matrix ))
        Xdense = X.tocsr()
