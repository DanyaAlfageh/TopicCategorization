import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrix():

    def __init__(self, yPredicted, yActual):
        self.yPredicted = yPredicted
        self.yActual = yActual
        self.matrix = np.zeros((20, 20))
        print(self.matrix)
        self.labels = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                       'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
                       'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
                       'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
        self.make_confusion_matrix()
        self.show_confusion_matrix()

    def make_confusion_matrix(self):
        for pred, act in zip(self.yPredicted, self.yActual):
            self.matrix[pred-1][act-1] += 1
        print(self.matrix)

    def show_confusion_matrix(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.matrix)
        ax.set_xticks(np.arange(20))
        ax.set_yticks(np.arange(20))
        ax.set_xticklabels(self.labels)
        ax.set_yticklabels(self.labels)

        plt.setp(ax.get_xticklabels(), rotation =45, ha='right', rotation_mode='anchor')
        for i in range(20):
            for j in range(20):
                if i == j:
                    color = 'b'
                else:
                    color = 'w'
                text = ax.text(j, i, self.matrix[i, j], ha='center', va='center', color=color)

        ax.set_title('ConfusionMatrix')
        plt.show()
