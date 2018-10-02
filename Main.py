import sys
from IO import DataIn,DataOut
from Naive import NaiveBayes
from Regression import LinearRegression
from Vocabulary import Vocabulary

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
    vocab = Vocabulary()
    print("Building ML model..")
    if (self.mode == 'naive'):
        naive = NaiveBayes(self.trainingData, self.testingData, self.predictionData, vocab)
    if (self.mode == 'regression'):
        regression = LinearRegression(self.trainingData, self.testingData, self.predictionData,vocab)
        regression.make_delta_matrix()
    print("Prediction available in /data/prediction.csv")


  """
    Working on the necessary conversion of the sparce Matrix
  """
  def prep_data(self):
    training = DataIn(file = 'training')
    testing = DataIn(file = 'testing')
    self.trainingData = training.load_dense_matrix()
    self.testingData = testing.load_dense_matrix()
    prediction = DataIn(file = 'testing')
    self.predictionData = prediction

  """
   Verifies the command line arguments,calls for
   print usage if invalid command line args.
  """
  def command_line_args(self):
    argLength = len(sys.argv)
    if(argLength == 3):

      #decision of which algorithm to use
      function = sys.argv[1].lower()
      if(function == 'regression' or function == 'naive'):
        self.mode = function
      else:
        self.print_usage()
        exit(1)

      #Are we going to cache the data?
      cache = sys.argv[2].lower()
      if(cache == 'y' or cache == 'n'):
        if(cache == 'y'): self.cache = True
        if(cache == 'n'): self.cache = False
        print(self.cache)
      else:
        self.print_usage()
        exit(1)


    #wrong amount of command line args.
    else:
      self.print_usage()
      exit(1)

  """
    prints the usage for command line arguments
    in case wrong kinds of args are given
  """
  def print_usage(self):
    print("Usage:")
    print("ALGORITHM CACHE")
    print("Algorithm: 'native' (Naive Bayes) or 'regression' (Logistic Regression)")
    print("CACHE: 'y' or 'n'.")
    print("       If data set will be constant, use Y to use cached static data attributes.")

main = Main()
