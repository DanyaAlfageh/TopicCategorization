import random

indices = set(list(range(12000)))
validationIndicies = set(random.sample(indices, 2400)) #20 percent of 12000 samples into validation set
trainingIndicies = indices - validationIndicies
viList = list(validationIndicies)
tiList = list(trainingIndicies)
print('Trying to write out training and validation')
try:
    with open('validationIndiciesNew.txt') as f:
        for index in viList:
            f.write(str(index) + "\n")
except:
    print('What is wrong')
print('wrote validationIndicies')

with open('trainingIndiciesNew.txt') as t:
    for index in tiList:
        t.write(str(index) + "\n")
print('wrote trainingIndicies')
