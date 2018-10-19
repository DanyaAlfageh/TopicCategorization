import csv
import sys
import random
import numpy as np
import pandas as pd
import scipy.sparse as ss
from collections import Counter
from Confusion import ConfusionMatrix

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


    """
    Creates a dense representation of the testing set
    """
    def create_dense_matrix(self, file):
        #Excluding first column, which containes ids
        cols = list(range(61189))
        colsToUse = cols[1:]
        print('starting to read csv of trainingData')
        #Reading in a pandas dataFrame
        data = pd.read_csv('data/sparse/testing.csv', sep=',', header=None, dtype=np.float64, usecols=colsToUse)
        print('finished reading csv')
        #Converting to csr matrix
        sparseCoo = ss.coo_matrix(data)
        sparse = sparseCoo.tocsr()
        print('SUCCESS: cONVERTING FROM DATAFRAME TO SPARSE MATRIX')
        attributes = {
            'data': sparse.data,
            'indices': sparse.indices,
            'indptr': sparse.indptr,
            'shape': sparse.shape
        }
        #Writing out newly created representation to file
        np.savez('data/dense/denseRepTesting.npz', **attributes)
        print('sparse testing saved')

    """
    Creating the dense represntations of the validation and training sets
    """
    def create_dense_matrix_training_and_validation(self,file):
        try:
            cols = list(range(61190))
            colsToUse = cols[1:]
            #Index list for validation and training sets
            viList = []
            tiList = []
            iList = []
            #Adding validation indices to list
            vi = open('data/validationIndicies.txt', 'r')
            for line in vi:
                viList.append(int(line))
            #Adding training indices to list
            ti = open('data/trainingIndicies.txt', 'r')
            for line1 in ti:
                tiList.append(int(line1))
            iList.append(viList)
            iList.append(tiList)
            test = 0
            #Creating both the trainng and validation dense representations
            for i in iList:
                print('starting to read csv for training and validation')
                data = pd.read_csv('data/sparse/training.csv', sep=',', header=None, dtype=np.float64, usecols=colsToUse, skiprows=i)
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
        except:
            print("[Error]: No file found at '"+file+"'.")

    def create_single_dense_matrix(self,id):
        final = []
        matrix = self.load_training_matrix().tolil()
        classes = matrix.getcol(matrix.shape[1]-1)
        for x in reversed(range(matrix.shape[0])):
            if(classes[x] != id):
                self.delete_row_lil(matrix, x)
        matrix = matrix.tocsr()
        for x in range(0,matrix.shape[1]-1):
          final.append(matrix.getcol(x).sum())
        #final.append(id)
        preConvert = np.array(final)
        matrix = ss.csr_matrix(preConvert)
        attributes = {
          'data': matrix.data,
          'indices': matrix.indices,
          'indptr': matrix.indptr,
          'shape': matrix.shape
        }
        np.savez('data/dense/class/'+str(id)+'.npz', **attributes)

    def create_naive_bayes_matrix(self):
      fullBayesMatrix = self.load_single_dense_matrix(1)
      for x in range(1,21):
          fullBayesMatrix = ss.vstack([fullBayesMatrix,self.load_single_dense_matrix(x)])
      attributes = {
        'data': fullBayesMatrix.data,
        'indices': fullBayesMatrix.indices,
        'indptr': fullBayesMatrix.indptr,
        'shape': fullBayesMatrix.shape
      }
      np.savez('data/dense/class/full.npz', **attributes)


    def delete_row_lil(self,mat, i):
      mat.rows = np.delete(mat.rows, i)
      mat.data = np.delete(mat.data, i)
      mat._shape = (mat._shape[0] - 1, mat._shape[1])


    def load_dense_matrix(self):
        loader = np.load('data/dense/denseRepNoIDs.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])
        #print(matrix)
        return matrix

    def load_training_matrix(self):
        loader = np.load('data/dense/denseRepTraining.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])
        #print(matrix)
        return matrix

    def load_testing_matrix(self):
        loader = np.load('data/dense/denseRepTesting.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])
        #print(matrix)
        return matrix

    def load_validation_matrix(self):
        loader = np.load('data/dense/denseRepValidation.npz')
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

    def load_naive_bayes_matrix(self):
        loader = np.load('data/dense/class/full.npz')
        args = (loader['data'], loader['indices'], loader['indptr'])
        matrix = ss.csr_matrix(args, shape=loader['shape'])
        return matrix

    #return the list version of the input data
    def get_whole_list(self):
      return self.lines



"""
A wrapper around the CSV code for our purposes.
Writes output to given file name.
"""
class DataOut():

    def __init__(self, fileName= 'data/prediction.csv'):
        self.lines = []
        self.fileName = fileName
    def add(self, id, classification):
        self.lines.append([id,classification])

    def write(self):
      with open(self.fileName, "w+") as csvFile:
        fileWriter = csv.writer(csvFile, delimiter=',')
        fileWriter.writerow(["id","class"]) #standard header
        for line in self.lines:
            fileWriter.writerow(line)

    def generate_confusion_matrix(self, correctList):
        confusion = ConfusionMatrix(self.lines, correctList)
