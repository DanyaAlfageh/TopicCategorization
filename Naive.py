from IO import CacheIn, CacheOut


class NaiveBayes():

  

    def __init__(self, trainingData, testingData, predictionData, cache = False):
        self.trainingData = trainingData
        self.testingData = testingData
        self.predictionData = predictionData
        self.cache = cache
        test = CacheIn('naive')
        print(test.get_cache_value(1))
        testTwo = CacheOut('naive')
        testTwo.add(1,2)
        testTwo.write()

       
