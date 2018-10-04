from collections import Counter
from multiprocessing import Pool
from Vocabulary import Vocabulary
from IO import CacheIn, CacheOut

class NaiveBayes():

  vocabListLength = -1;
  alphaMinusOne = -1;

  def __init__(self, trainingData, validationData, testingData):
        #One time overhead computation
        if(NaiveBayes.vocabListLength == -1):
            vocab = Vocabulary()
            NaiveBayes.vocabListLength = vocab.length
            NaiveBayes.alphaMinusOne = 1/vocab.length

        self.trainingData = trainingData
        self.validationData = validationData
        self.testingData = testingData
        self.columns = trainingData.shape[1] #Words - 1
        self.trainingRows = trainingData.shape[0] #classifications -1
        self.testingRows = testingData.shape[0]
        self.MLE = dict()
        self.calc_mle()

        pool = Pool(100)
        pool.map(self.get_MAP, [x for x in range(self.testingRows)])





  def calc_mle(self):
      Y = Counter(self.trainingData.getcol(self.columns-1).data)
      total = len(self.trainingData.getcol(self.columns-1).data)
      for k in Y:
        MLE = Y.get(k)/total
        self.MLE[k] = MLE

  def get_mle(self, Y):
      self.trainingData.getcol(self.columns-1).data
      return self.MLE[Y]

  def get_MAP(self, x):
        for j in range(0,self.trainingRows,1):
          row = self.trainingData.getrow(j).data
