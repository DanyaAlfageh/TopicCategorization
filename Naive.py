import math
import numpy as np
from scipy import sparse
from collections import Counter
from multiprocessing import Pool
from Vocabulary import Vocabulary
from IO import CacheIn, CacheOut

class NaiveBayes():

  def __init__(self, trainingData, validationData, testingData, naiveBayesMatrix):
        self.trainingData = trainingData
        self.validationData = validationData
        self.testingData = testingData
        self.columns = trainingData.shape[1] #Words - 1
        self.trainingRows = trainingData.shape[0] #classifications -1
        self.testingRows = testingData.shape[0]
        self.naiveBayesMatrix = naiveBayesMatrix
        self.MLE = self.calc_mle()
        print(self.MLE)
        map = Map_Matrix(naiveBayesMatrix)
        summation = Summation(map)



  def calc_mle(self):
      MLE = dict()
      Y = Counter(self.trainingData.getcol(self.columns-1).data)
      total = len(self.trainingData.getcol(self.columns-1).data)
      for k in Y:
        MLEk = Y.get(k)/total
        MLE[k] = MLEk
      return MLE

  def get_mle(self, Y):
      self.trainingData.getcol(self.columns-1).data
      return self.MLE[Y]


class Summation():

    def __init__(self, map):
        logMatrix = 1/math.log(2)* np.log(map.MAPMatrix) # (log2(P(Xi|Yk)))



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
    self.MAPMatrix = np.zeros(0)
