import math
import numpy as np
from scipy import sparse
from collections import Counter
from multiprocessing import Pool
from Vocabulary import Vocabulary
from IO import DataOut

"""
Uses the naive bayes classifier to determine the most accurate Classification
of data given prior example classifications and information about the newest
data to be classified. Uses linear algebra and a flattening of data to assist if
in a speedy calculation.
"""
class NaiveBayes():

  validating = False;
  def __init__(self, trainingData, validationData, testingData, naiveBayesMatrix, beta = -1):
        self.trainingData = trainingData
        if(NaiveBayes.validating): self.testingData = validationData
        else: self.testingData = testingData
        self.naiveBayesMatrix = naiveBayesMatrix
        out = DataOut()

        #MLE -> log2(P(Y))
        MLEMatrix = MLE_Matrix(trainingData)
        MLEMatrix = MLEMatrix.get_log_mle_matrix()
        #MAP -> log2(P(X|Y))
        mapMatrix = Map_Matrix(naiveBayesMatrix,beta)
        mapMatrix = mapMatrix.get().transpose()#so we can use linear algebra to multipy

        #Classification -> argmax[MLE + MAP]
        for x in range (0,self.testingData.shape[0]):
            currentRow = self.testingData.getrow(x).todense()
            if(NaiveBayes.validating):currentRow = np.delete(currentRow,self.testingData.shape[1]-1,1)
            results = currentRow.dot(mapMatrix)
            classification = np.argmax(results)
            out.add(x+12001,classification)
        if(NaiveBayes.validating):
           correct = self.testingData.getcol(self.testingData.shape[1]-1).data.astype(int).tolist()
           out.generate_confusion_matrix([x for x in correct])
        else:
           out.write()

"""
Constructs a matrix that holds the P(Y) relative
to Y in that column.
"""
class MLE_Matrix():

  def __init__(self,data):
      MLE = dict()
      data = data.getcol(data.shape[1]-1).data
      Y = Counter(data)
      total = len(data)
      for k in Y:
        MLEk = Y.get(k)/total
        MLE[k] = MLEk
      self.MLE = MLE

  def get(self, id):
      return self.MLE[id]

  def get_mle_matrix(self):
      matrix = np.zeros((20,1))
      for x in range(1,21):
          matrix[x-1] = self.MLE[x]
      return np.log2(matrix)

  def get_log_mle_matrix(self):
      return np.log2(self.get_mle_matrix())

"""
Calculates an MAP matrix that determines the P(X|Y)
for any X Y combination, then sets them in the corresponding Matrix
position where the original X was.
"""
class Map_Matrix():

  def __init__(self, naiveBayesMatrix,beta = -1):
    naiveBayesMatrix = naiveBayesMatrix.todense()

    v = 0 #total Vocabulary words
    for x in range(0,naiveBayesMatrix.shape[0]):
        v = v + naiveBayesMatrix[x,:].sum()

    #B = 1/v
    B = beta
    if(beta == -1): B = 1/v
    alphaMinusOne = B #(a-1)

    #(length of vocab list)
    vocab = Vocabulary()
    vocabListLength = vocab.length

    # (a-1)*(length of vocab list)
    denominatorStatic = alphaMinusOne * vocabListLength

    #(count of Xi in Yk) + (a-1)
    numerator = naiveBayesMatrix + alphaMinusOne

    #P(Xi|Yk)
    for x in range(numerator.shape[0]):
        denominatorDynamic = naiveBayesMatrix[x,:].sum()
        numerator[x,:] *= (1/(denominatorDynamic + denominatorStatic))

    #log2(P(Xi|Yk))
    self.mapmatrix = np.log2(numerator)

  def get(self):
      return self.mapmatrix
