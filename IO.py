import csv
import sys
import random
import numpy as np
import pandas as pd
import scipy.sparse as ss

# A wrapper around the CSV code for
# our purposes
class DataIn():

    lines = []

    #param dataSet -> training or testing
    def __init__(self,file = 'training', load_dense_matrix = True, path="data/sparse/"):
       if load_dense_matrix:
           print('Loading matrix')
           self.denseMatrix = self.load_dense_matrix()
           print('loaded dense respresentation correctly')
       else:
           try:
             with open(path+file+'.csv') as csvFile:
               temp = csv.reader(csvFile, delimiter=',')
               self.lines = list(temp)
               print("[Info]: File found for "+file+".")
           except:
              print("Creating sparse matrix without ID's")
              self.create_dense_matrix(file)
              print("[Warning]: No file Found for "+file+".")



    def create_dense_matrix(self, file):
        cols = list(range(61189))
        colsToUse = cols[1:]
        print('starting to read csv of testingData')
        data = pd.read_csv('data/sparse/testing.csv', sep=',', header=None, dtype=np.float32, usecols=colsToUse)
        print('finished reading csv')
        sparseCoo = ss.coo_matrix(data)
        sparse = sparseCoo.tocsr()
        print('SUCCESS: cONVERTING FROM DATAFRAME TO SPARSE MATRIX')
        attributes = {
            'data': sparse.data,
            'indices': sparse.indices,
            'indptr': sparse.indptr,
            'shape': sparse.shape
        }
        np.savez('data/dense/denseRepTesting.npz', **attributes)
        print('sparse testing saved')

    def create_dense_matrix_training_and_validation(self,file):
        try:
            cols = list(range(61190))
            colsToUse = cols[1:]
            viList = []
            tiList = []
            iList = []
            vi = open('data/validationIndicies.txt', 'r')
            for line in vi:
                viList.append(int(line))
            ti = open('data/trainingIndicies.txt', 'r')
            for line1 in ti:
                tiList.append(int(line1))
            iList.append(viList)
            iList.append(tiList)
            test = 0
            for i in iList:
                print('starting to read csv')
                data = pd.read_csv('data/sparse/training.csv', sep=',', header=None, dtype=np.float32, usecols=colsToUse, skiprows=i)
                print('finished reading csv')
                sparseCoo = ss.coo_matrix(data)
                sparse = sparseCoo.tocsr()
                print('SUCCESS: cONVERTING FROM DATAFRAME TO SPARSE MATRIX')
                attributes = {
                    'data': sparse.data,
                    'indices': sparse.indices,
                    'indptr': sparse.indptr,
                    'shape': sparse.shape
                }
                if test == 0:
                    name = 'Training'
                else:
                    name = 'Validation'
                np.savez('data/dense/denseRep'+name+'.npz', **attributes)
                test += 1

        except Exception as e:
          print(e)
          print("[Error]: No file found at '"+file+"'.")

    def load_dense_matrix(self):
        loader = np.load('data/dense/denseRepNoIDs.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])
        #print(matrix)
        return matrix

    def create_training_and_validation(self):
        indices = set(list(range(12000)))
        validationIndicies = set(random.sample(indices, 2400)) #20 percent of 12000 samples into validation set
        trainingIndicies = indices - validationIndicies

        print('set difference worked')

    #return the list version of the input data
    def get_whole_list(self):
      return self.lines



# A wrapper around the CSV code for
# our purposes
class DataOut():

    def __init__(self):
        self.lines = []

    def add(self, id, classification):
        self.lines.append([id,classification])

    def write(self):
      with open('data/prediction.csv', "w+") as csvFile:
        fileWriter = csv.writer(csvFile, delimiter=',')
        fileWriter.writerow(["id","class"]) #standard header
        for line in self.lines:
            fileWriter.writerow(line)



class CacheIn():

    cache = []
    mode = ''

    def __init__(self, mode):
     print("Loading in "+mode+" Cache...")
     try:
      exists = open(mode+'/cache.csv')
      with open(mode+'/cache.csv') as csvFile:
        temp = csv.reader(csvFile, delimiter=',')
        self.cache = list(temp)
     except:
       print("No "+mode+" Cache Found.")

    def cache_exists(self):
      if not self.cache:
        return False
      return True

    def get_cache_value(self, key):
      if not self.cache:
        raise Exception('No Cache was found.')
        exit(1)
      for pairs in self.cache:
       print(pairs)
       if (pairs[0] == str(key)):
         return pairs[1]
      return -1


class CacheOut():

    lines = []
    mode = ''

    def __init__(self,mode):
      self.mode = mode

    def add(self, key, value):
        self.lines.append([key,value])

    def write(self):
      with open(self.mode+'/cache.csv', "w+") as csvFile:
        fileWriter = csv.writer(csvFile, delimiter=',')
        fileWriter.writerow(["key","value"]) #standard header
        for line in self.lines:
            fileWriter.writerow(line)
