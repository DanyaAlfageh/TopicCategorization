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
           self.denseMatrix = self.load_dense_matrix()
           print('loaded dense respresentation correctly')
           #print(self.denseMatrix[1])
       else:
           try:
             with open(path+file+'.csv') as csvFile:
               temp = csv.reader(csvFile, delimiter=',')
               self.lines = list(temp)
               print("[Info]: File found for "+file+".")
           except:
             self.create_dense_matrix(file)
             print("[Warning]: No file Found for "+file+".")


    def create_dense_matrix(self,file):
        try:
            exists = open(path+file+'.csv')
            #data = np.generatefromtxt('data/sparse/'+file+'.csv',delimiter = ',')
            data = pd.read_csv(path+file+'.csv', sep=',', header=None, dtype=np.uint32)

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
          print("[Error]: No file found at '"+file+"'.")

    def create_single_dense_matrix(self,id):
        final = []
        matrix = self.load_dense_matrix().tolil()
        classes = matrix.getcol(matrix.shape[1]-1)
        for x in reversed(range(matrix.shape[0])):
            if(classes[x] != id):
                self.delete_row_lil(matrix, x)
        matrix = matrix.tocsr()
        for x in range(matrix.shape[1]-1):
          final.append(matrix.getcol(x).sum())
        final.append(id)
        preConvert = np.array(final)
        matrix = ss.csr_matrix(preConvert)
        attributes = {
          'data': matrix.data,
          'indices': matrix.indices,
          'indptr': matrix.indptr,
          'shape': matrix.shape
        }
        np.savez('data/dense/class/'+str(id)+'.npz', **attributes)

    def delete_row_lil(self,mat, i):
      mat.rows = np.delete(mat.rows, i)
      mat.data = np.delete(mat.data, i)
      mat._shape = (mat._shape[0] - 1, mat._shape[1])


    def load_dense_matrix(self):
        loader = np.load('data/dense/denseRep.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])
        #print(matrix)
        return matrix

    def load_single_dense_matrix(self,id):
        loader = np.load('data/dense/class/'+str(id)+'.npz')
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
