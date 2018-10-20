import math
from Naive import Map_Matrix, MLE_Matrix


class MutualInformation():


    def __init__(self, data, naiveBayesMatrix):
        final = []
        mleP = MLE_Matrix(data)
        map = Map_Matrix(naiveBayesMatrix)
        mapP = map.get()
        px = ProbablityX(naiveBayesMatrix);
        for y in range(0,naiveBayesMatrix.shape[1]):
            IG = 0
            for x in range(0,naiveBayesMatrix.shape[0]):
              mle = 0
              if(x == 0): mle = mleP.get(1)
              else: mle = mleP.get(x)
              pxy = abs(mapP[x,y])* mle
              temp =math.log(pxy / (px.get(y) * mle))
              IG = IG + temp
            final.append((y,IG))
            print(y)
        with open('ranking.txt', 'w') as fp:
          fp.write('\n'.join('%s %s' % x for x in final))


class ProbablityX():

    def __init__(self,naiveBayesMatrix):
      v = 0 #total Vocabulary words
      countX = []
      for x in range(0,naiveBayesMatrix.shape[0]):
          v = v + naiveBayesMatrix[x,:].sum()

      for y in range(0,naiveBayesMatrix.shape[1]):
          countX.append(naiveBayesMatrix[:,y].sum())

      self.countX = [x * (1/v) for x in countX]

    def get(self, x):
        return self.countX[x]
