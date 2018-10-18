import math
import numpy as np
from scipy import sparse
from collections import Counter
from multiprocessing import Pool
from Vocabulary import Vocabulary
from IO import CacheIn, CacheOut, DataOut

"""
Uses the naive bayes classifier to determine the most accurate Classification
of data given prior example classifications and information about the newest
data to be classified.
"""
class NaiveBayes():

  """

  """
  def __init__(self, trainingData, validationData, testingData, naiveBayesMatrix):
        self.trainingData = trainingData
        self.validationData = validationData
        self.testingData = testingData
        self.naiveBayesMatrix = naiveBayesMatrix
        out = DataOut()

        #MLE -> log2(P(Y))
        MLEMatrix = self.get_mle_matrix()

        #MAP -> log2(P(X|Y))
        mapMatrix = Map_Matrix(naiveBayesMatrix)
        mapMatrix = mapMatrix.get().transpose() #so we can use linear algebra to multipy


        #Classification -> argmax[MLE + MAP]
        for x in range (0,testingData.shape[0]):
            currentRow = testingData.getrow(x)
            results = (currentRow * mapMatrix)
            #print(results)
            classification = np.argmax(results)
            out.add(x+12001,classification)
        out.write()

  def calc_mle(self):
      MLE = dict()
      data = self.trainingData.getcol(self.trainingData.shape[1]-1).data
      Y = Counter(data)
      total = len(data)
      for k in Y:
        MLEk = Y.get(k)/total
        MLE[k] = MLEk
      return MLE

  def get_mle_matrix(self):
      MLE = self.calc_mle()
      matrix = np.zeros((20,1))
      for x in range(1,21):
          matrix[x-1] = MLE[x]
      return matrix

class Map_Matrix():

  def __init__(self, naiveBayesMatrix):
    naiveBayesMatrix = naiveBayesMatrix.todense()
    print("Here")
    print(naiveBayesMatrix)
    print(naiveBayesMatrix.shape[1])
    naiveBayesMatrix = np.delete(naiveBayesMatrix,0,1)
    #naiveBayesMatrix = np.delete(naiveBayesMatrix,naiveBayesMatrix.shape[1]-1,1)

    v = 0 #total Vocabulary words
    for x in range(0,naiveBayesMatrix.shape[0]):
        v = v + naiveBayesMatrix[x,:].sum()

    #B = 1/v
    B = 1/v

    vocab = Vocabulary()
    vocabListLength = vocab.length #(length of vocab list)
    alphaMinusOne = B #(a-1)

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
