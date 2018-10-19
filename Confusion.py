import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np

"""
Class used to create confusion matrices from the results of Naive Bayes and Logistic Regression
Arguements are a list of predicted classes and a list of actual classes
"""
class ConfusionMatrix():

    def __init__(self, yPredicted, yActual):
        self.yPredicted = yPredicted
        self.yActual = yActual
        #matrix to represent the confusion matrix
        self.matrix = np.zeros((20, 20))
        #Labels of the 20 different classes
        self.labels = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                       'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
                       'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
                       'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
        self.make_confusion_matrix()
        self.calculate_error_percentage()
        self.show_confusion_matrix()

    """
    Iterates through both lists and adds one to the count at row = pred, col = act
    """
    def make_confusion_matrix(self):
        for pred, act in zip(self.yPredicted, self.yActual):
            print("Here")
            self.matrix[pred-1][act-1] += 1

    """
    Creates the visualzation of the confusion matrix
    """
    def show_confusion_matrix(self):
        #Creating the figure and the axis
        fig, ax = plt.subplots()
        im = ax.imshow(self.matrix)
        #Setting the ticks and labels for accuracy
        ax.set_xticks(np.arange(20))
        ax.set_yticks(np.arange(20))
        ax.set_xticklabels(self.labels)
        ax.set_yticklabels(self.labels)
        plt.setp(ax.get_xticklabels(), rotation =45, ha='right', rotation_mode='anchor')
        #Setting the text for each entry of matrix.
        for i in range(20):
            for j in range(20):
                if i == j:
                    color = 'b'
                else:
                    color = 'w'
                text = ax.text(j, i, self.matrix[i, j], ha='center', va='center', color=color)

        ax.set_title('ConfusionMatrix')
        plt.show()

    """
    Calculates the percentage of examples that were misclassified as each class.
    """
    def calculate_error_percentage(self):
        for i in range(20):
            errors = 0
            total = 0
            for j in range(20):
                matVal = self.matrix[i][j]
                total += matVal
                if i != j:
                    errors += matVal
            print('Row', i, 'has error percentage of', errors/total, 'with total', total)
