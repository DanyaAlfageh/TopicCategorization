import math
import numpy as np
from scipy import sparse
from collections import Counter
from multiprocessing import Pool
from Vocabulary import Vocabulary
from IO import CacheIn, CacheOut, DataOut

class NaiveBayes():

  def __init__(self, trainingData, validationData, testingData, naiveBayesMatrix):
        self.trainingData = trainingData
        self.validationData = validationData
        self.testingData = testingData
        self.columns = trainingData.shape[1] #Words - 1
        self.trainingRows = trainingData.shape[0] #classifications -1
        self.testingRows = testingData.shape[0]

        out = DataOut()
        self.naiveBayesMatrix = naiveBayesMatrix
        self.MLE = self.calc_mle()
        mapMatrix = Map_Matrix(naiveBayesMatrix)
        map = mapMatrix.MAPMatrix.transpose()
        MLEMatrix = self.get_mle_matrix()
        print(testingData.shape[0])
        for x in range (0,testingData.shape[0]):
            currentRow = testingData.getrow(x).todense()
            currentRow = np.delete(currentRow,0,1)
            results = (currentRow * map)
            classification = np.argmax(results)
            out.add(12001+x,classification)
        out.write()






  def calc_mle(self):
      MLE = dict()
      Y = Counter(self.trainingData.getcol(self.columns-1).data)
      total = len(self.trainingData.getcol(self.columns-1).data)
      for k in Y:
        MLEk = Y.get(k)/total
        MLE[k] = MLEk
      return MLE

  def get_mle_matrix(self):
      matrix = np.zeros((20,1))
      for x in range(1,21):
          matrix[x-1] = self.MLE[x]
      print(matrix)
      return matrix





class Map_Matrix():

  vocabListLength = -1;
  alphaMinusOne = -1;

  def __init__(self, naiveBayesMatrix):
    #One time overhead computation
    if(Map_Matrix.vocabListLength == -1):
      vocab = Vocabulary()
      Map_Matrix.vocabListLength = vocab.length
      Map_Matrix.alphaMinusOne = 1/vocab.length #(1/|v|)

    #self.denominator = totalWords + (Map_Matrix.alphaMinusOne * Map_Matrix.vocabListLength)
    alphaMatrix = np.full((21,61190), Map_Matrix.alphaMinusOne) #(a -1)
    self.numerator = naiveBayesMatrix + alphaMatrix #(count of Xi in Yk)+(a-1)
    denominatorStatic = Map_Matrix.alphaMinusOne * Map_Matrix.vocabListLength # (a-1)*(length of vocab list)
    for x in range(self.numerator.shape[0]):
        denominatorDynamic = self.numerator[x,:].sum() - (self.numerator[x,0] + self.numerator[x,self.numerator.shape[0]-1]) #(total words in Yk)
        self.numerator[x,:] *= (1/(denominatorDynamic + denominatorStatic))
        self.numerator[x,0] = 1
        self.numerator[x,self.numerator.shape[1]-1] = (x+1)
    self.MAPMatrix = self.numerator
    self.MAPMatrix = np.delete(self.MAPMatrix,0,1)
    self.MAPMatrix = np.delete(self.MAPMatrix,self.MAPMatrix.shape[1]-1,1)
    self.logMatrix = np.log2(self.MAPMatrix) # (log2(P(Xi|Yk)))\
    self.MAPMatrix = self.logMatrix
