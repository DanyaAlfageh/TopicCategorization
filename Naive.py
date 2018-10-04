from collections import Counter
from Vocabulary import Vocabulary
from IO import CacheIn, CacheOut

class NaiveBayes():

  vocabListLength = -1;
  alphaMinusOne = -1;

  def __init__(self, trainingData, testingData, predictionData):
        #One time overhead computation
        if(NaiveBayes.vocabListLength == -1):
            vocab = Vocabulary()
            NaiveBayes.vocabListLength = vocab.length
            NaiveBayes.alphaMinusOne = 1/vocab.length

        self.trainingData = trainingData
        self.testingData = testingData
        self.predictionData = predictionData
        self.columns = trainingData.shape[1] #Words - 1
        self.rows = trainingData.shape[0] #classifications -1
        self.MLE = dict()
        self.calc_mle()

        i = 0
        while i < self.rows:
            self.trainingData.getrow(self.rows-1).data
            i = i +1


  def calc_mle(self):
      Y = Counter(self.trainingData.getcol(self.columns-1).data)
      total = len(self.trainingData.getcol(self.columns-1).data)
      for k in Y:
        MLE = Y.get(k)/total
        self.MLE[k] = MLE

  def get_mle(self, Y):
      self.trainingData.getcol(self.columns-1).data
      return self.MLE[Y]
