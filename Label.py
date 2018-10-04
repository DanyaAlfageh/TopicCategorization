from Vocabulary import Vocabulary


class Label():

  totalDocs = 12000
  vocabListLength = -1;
  alphaMinusOne = -1;


  def __init__(self, Yk, MLE ,dataSet):
    if(Label.vocabListLength == -1):
        vocab = Vocabulary()
        Label.vocabListLength = vocab.length
        Label.alphaMinusOne = 1/vocab.length
    self.label = Yk
    self.MLE = MLE
    self.dataSet = dataSet;

  def get_MLE():
     return self.MLE

  def get_MAP(xi):
     return -1
