from collections import Counter
from IO import CacheIn, CacheOut
from Label import Label


class NaiveBayes():
    docsCol = 61189
    labels = list()


    def __init__(self, trainingData, testingData, predictionData):
        self.trainingData = trainingData
        self.testingData = testingData
        self.predictionData = predictionData
        self.load_labels()


    def load_labels(self):
      Y = Counter(self.trainingData.getcol(self.docsCol).data)
      print(Y)
      total = len(self.trainingData.getcol(self.docsCol).data)
      for k in Y:
        MLE = Y.get(k)/total
        print(""+str(MLE) +" "+ str(Y.get(k))+" "+str(total))
        self.labels.append(Label(k,MLE,self.trainingData))
