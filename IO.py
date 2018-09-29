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
    def __init__(self,file = 'training', load_dense_matrix = True):
       if load_dense_matrix:
           self.denseMatrix = self.load_dense_matrix()
           print('loaded dense respresentation correctly')
           print(self.denseMatrix[0])
       else:
           try:
             with open('data/dense/'+file+'.csv') as csvFile:
               temp = csv.reader(csvFile, delimiter=',')
               self.lines = list(temp)
               print("[Info]: Dense Matrix found for "+file+".")
           except:
             self.create_dense_matrix(file)
             print("[Warning]: No Dense Matrix Found for "+file+".")
             print("[Info]: Generating dense matrix.")


    def create_dense_matrix(self,file):
        try:
            exists = open('data/sparse/'+file+'.csv')
            #data = np.generatefromtxt('data/sparse/'+file+'.csv',delimiter = ',')
            data = pd.read_csv('data/sparse/'+file+'.csv', sep=',', header=None, dtype=np.uint32)
            print('finished reading csv')
            rows = data[0]
            cols = data[1]
            print(data)
            print("Testing rows ")
            print(rows[0])
            print('numRows: ', len(rows), ' numCols: ', len(cols))
            sparseCoo = ss.coo_matrix(data)
            sparse = sparseCoo.tocsr()
            print('SUCCESS: cONVERTING FROM DATAFRAME TO SPARSE MATRIX')
            attributes = {
                'data': sparse.data,
                'indices': sparse.indices,
                'indptr': sparse.indptr,
                'shape': sparse.shape
            }
            np.savez('data/dense/denseRep.npz', **attributes)
        except:
          print("[Error]: No file found at 'data/sparse/"+file+"'.")

    def load_dense_matrix(self):
        loader = np.load('data/dense/denseRep.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])
        return matrix

    def create_training_and_validation(self):
        indices = set(list(range(1200)))
        validationIndicies = set(random.sample(indices, 240)) #20 percent of 1200 samples into validation set
        trainingIndicies = indices - validationIndicies
        print('set difference worked')




# A wrapper around the CSV code for
# our purposes
class DataOut():

    lines = []

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
