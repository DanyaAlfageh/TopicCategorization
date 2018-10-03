from IO import DataIn

"""
A representation of the vocabulary.txt file that is called upon
to be used as a virtual cache.
"""
class Vocabulary():

    vocab = [] #the vocab list itself.

    def __init__(self):
        vocabIn = DataIn(file="vocabulary",load_dense_matrix=False, path="data/")
        #https://stackoverflow.com/questions/11264684/flatten-list-of-lists/11264799
        self.vocab = [val for sublist in vocabIn.get_whole_list() for val in sublist]

    """
     Grabs the index relative to the word in the vocab list
     Adds +1 to account for ID offset in dataset.
    """
    def get_index(self, word):
      try:
        return list.index(word) +1
      except ValueError:
        return -1;

    def get_word(self, index):
        return vocab[index-1]