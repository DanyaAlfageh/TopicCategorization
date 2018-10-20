import sys
from IO import DataIn,DataOut
from Naive import NaiveBayes
from Regression import LinearRegression
from Vocabulary import Vocabulary
from Confusion import ConfusionMatrix

"""
  The main class, running the show and
  running the higher level direction of the program
"""
class Main():

  """
    Calls the verification of the command line args,
    then selects the algorithm to run
  """
  def __init__(self):
    self.command_line_args()
    self.prep_data()
    print("Building ML model..")
    if (self.mode == 'naive'):
        naive = NaiveBayes(self.trainingData, self.validationData, self.testingData,self.naiveBayesMatrix)
    if (self.mode == 'regression'):
        regression = LinearRegression()
        #regression.classifyData(fileName ='weightsLR0.0105PT0.015iters17001.npz', validation =True, createConfusion =True)
        #regression.find_score_for_all_computed_weights()
    if (self.mode == 'beta'):
        #for x in self.frange(.00001, 1.1, .2):
            #naive = NaiveBayes(self.trainingData, self.validationData, self.testingData,self.naiveBayesMatrix, beta = x)
        naive = NaiveBayes(self.trainingData, self.validationData, self.testingData,self.naiveBayesMatrix, beta = 1)
    print("Prediction available in /data/prediction.csv")


  """
    Working on the necessary conversion of the sparce Matrix
  """
  def prep_data(self):
    training = DataIn(file = 'training')
    testing = DataIn(file = 'testing')
    self.trainingData = training.load_training_matrix()
    #for x in range(1,21):
    #  training.create_single_dense_matrix(x)
    #training.create_naive_bayes_matrix()
    self.naiveBayesMatrix = training.load_naive_bayes_matrix()
    self.validationData = testing.load_validation_matrix()
    self.testingData = testing.load_testing_matrix()

  """
   Verifies the command line arguments,calls for
   print usage if invalid command line args.
  """
  def command_line_args(self):
    argLength = len(sys.argv)
    if(argLength == 2):

      #decision of which algorithm to use
      function = sys.argv[1].lower()
      if(function == 'regression' or function == 'naive' or function == 'beta'):
        self.mode = function
      else:
        self.print_usage()
        exit(1)

    #wrong amount of command line args.
    else:
      self.print_usage()
      exit(1)


  def frange(self,start, stop, step):
    i = start
    while i < stop:
      yield i
      i += step

  """
    prints the usage for command line arguments
    in case wrong kinds of args are given
  """
  def print_usage(self):
    print("Usage:")
    print("ALGORITHM CACHE")
    print("Algorithm: 'native' (Naive Bayes) or 'regression' (Logistic Regression)")

main = Main()
