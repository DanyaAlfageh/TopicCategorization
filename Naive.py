from collections import Counter
from IO import CacheIn, CacheOut


class NaiveBayes():
    docs = 61189
    Y = dict()
    def __init__(self, trainingData, testingData, predictionData, vocab):
        self.trainingData = trainingData
        self.testingData = testingData
        self.predictionData = predictionData
        Y = Counter(trainingData.getcol(self.docs).data)
        #print(Y)
