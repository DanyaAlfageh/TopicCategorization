import csv
import random
import numpy as np

# A wrapper around the CSV code for
# our purposes
class DataIn():

    lines = []

    #param dataSet -> training or testing
    def __init__(self,file = 'training'):
       try:
         with open('data/dense/'+file+'.csv') as csvFile:
           temp = csv.reader(csvFile, delimiter=',')
           self.lines = list(temp)
           print("[Info]: Dense Matrix found for "+file+".")
       except:
         self.create_dense_matrix()
         print("[Warning]: No Dense Matrix Found for "+file+".")
         print("[Info]: Generating dense matrix.")


    def create_dense_matrix():
        try:
            exists = open('data/sparse/'+file+'.csv')
            data = np.generatefromtxt('data/sparse/'+file+'.csv',delimiter = ',')
        except:
          print("[Error]: No file found at 'data/sparse/"+file+"'.")


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
