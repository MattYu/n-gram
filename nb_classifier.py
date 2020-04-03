from collections import defaultdict
import numpy as np
import math as math

from ngram import NGram

class MetaStatistics:

    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    f_measure = 0.0
    weighted_f_measure = 0.0
    beta = 1.0

    _correctDeductionsCount = 0
    _totalDeductionsCount = 0

class NaiveBayerClassifier(MetaStatistics):

    LANGUAGES = ['eu', 'ca', 'gl', 'es', 'en', 'pt']

    def __init__(self, *args, **kwargs):
        self.N = kwargs.pop('N', 3)
        self.V = kwargs.pop('V', 1)
        self.weight = kwargs.pop("weight", 0.001)
        if "LANGUAGES" in kwargs:
            self.LANGUAGES = kwargs.pop('LANGUAGES')
    

        self.beta = kwargs.pop("beta", 1)
        self.LanToNGrams= {}
        self.LanToIndex = {}
        self.test = kwargs.pop("test", False)

        self._precision_per_class = np.zeros(len(self.LANGUAGES))
        self._recall_per_class = np.zeros(len(self.LANGUAGES))
        self.confusion_matrix = np.full((len(self.LANGUAGES), len(self.LANGUAGES)), 0)

        self._TP_per_class = np.zeros(len(self.LANGUAGES))
        self._FP_per_class = np.zeros(len(self.LANGUAGES))
        self._FN_per_class = np.zeros(len(self.LANGUAGES))
        self._TN_per_class = np.zeros(len(self.LANGUAGES))

        self._f_measure_per_class =  np.zeros(len(self.LANGUAGES))


        self.count_per_class = np.zeros(len(self.LANGUAGES))
        self.totalCount = 0

        self.probability_per_language = np.zeros(len(self.LANGUAGES))

        index = 0
        for lan in self.LANGUAGES:
            
            self.LanToNGrams[lan] = NGram(lan = lan, N = self.N, V =self.V, weight = self.weight, test=self.test)
            self.LanToIndex[lan] = index
            index +=1

    def dynamicallyAddNewLanguage(self, lan):
        self.LANGUAGES.append(lan)
        self.LanToNGrams[lan] = NGram(lan = lan, N = self.N, V =self.V, weight = self.weight)
        self.LanToIndex[lan] = len(self.LanToIndex)
        newRow = [[0 for i in range(0, len(self.confusion_matrix[0]))]]
        self.confusion_matrix = np.append(self.confusion_matrix, newRow, 0)
        newCol = [[0] for i in range(0, len(self.confusion_matrix))]

        self.confusion_matrix = np.append(self.confusion_matrix, newCol, 1)

        while len(self._TP_per_class) != len(self.LANGUAGES):
            self._precision_per_class = np.hstack([self._precision_per_class, np.array([0.0])])
            self._recall_per_class = np.hstack([self._recall_per_class, np.array([0.0])])
            self._TP_per_class = np.hstack([self._TP_per_class, np.array([0.0])])
            self._FP_per_class = np.hstack([self._FP_per_class, np.array([0.0])])
            self._FN_per_class = np.hstack([self._FN_per_class, np.array([0.0])])
            self._TN_per_class = np.hstack([self._TN_per_class, np.array([0.0])])
            self.probability_per_language = np.hstack([self.probability_per_language, np.array([0.0])])


    def addTrainingInputLine(self, stringLine):
        lineList = stringLine.split("\t")
        #id = lineList[0]
        correctLan = lineList[2]
        txt = lineList[3]

        #print(id)
        #print(correctLan)
        #print(txt)

        if correctLan in self.LanToNGrams:
            self.LanToNGrams[correctLan].addTrainingInput(txt)
        else:
            self.dynamicallyAddNewLanguage(correctLan)
            self.LanToNGrams[correctLan].addTrainingInput(txt)

        self.count_per_class[self.LanToIndex[correctLan]] += 1
        self.totalCount += 1

    def trainFromTweets(self, fileName, maxLine = None):

        self.count_per_class = np.zeros(len(self.LANGUAGES))
        self.totalCount = 0

        self.probability_per_language = np.zeros(len(self.LANGUAGES))

        with open(fileName, encoding="utf8") as f:
            fil = f.read().splitlines()

            if not maxLine:
                for line in fil:
                    self.addTrainingInputLine(line)
            else:
                for line in fil[:min(len(fil), maxLine)]:
                    self.addTrainingInputLine(line)

        for lan in self.LANGUAGES:
            self.probability_per_language[self.LanToIndex[lan]] = self.count_per_class[self.LanToIndex[lan]]/self.totalCount


    def guessTweetLanguage(self, stringLine, logFile= None, newN= None, newWeight = None):
        lineList = stringLine.split("\t")
        if len(lineList) < 2:
            return
        id = lineList[0]
        correctLan = lineList[2]
        txt = lineList[3]
        #print(id)
        #print(correctLan)
        #print(txt)

        if correctLan not in self.LanToIndex:
            self.dynamicallyAddNewLanguage(correctLan)

        guesses = []

        for lan in self.LanToNGrams:
            lanProbability = self.probability_per_language[self.LanToIndex[lan]]
            if lanProbability == 0:
                lanProbability = 0.00000000000001
            guesses.append([lan, self.LanToNGrams[lan].getProbability(txt, newN, newWeight) + math.log10(lanProbability)])

        
        guesses.sort(key= lambda x: x[1], reverse = True)

        guessLan = guesses[0][0] 
        printStr = id + "  " + correctLan + "  " + str(guesses[0][1]) + "  " + guesses[0][0] + "  "

        self._totalDeductionsCount += 1

        guessIndex = self.LanToIndex[guesses[0][0]]
        realIndex = self.LanToIndex[correctLan]

        if guesses[0][0] == correctLan:
            printStr += "correct"
            self._TP_per_class[guessIndex] += 1
            self._correctDeductionsCount += 1
        else:
            printStr += "wrong"
            self._FP_per_class[guessIndex] += 1
            self._FN_per_class[realIndex] +=1

        self.confusion_matrix[realIndex, guessIndex] +=1
        #print(printStr)

    def runML(self, fileName, newN= None, newWeight = None, maxLine = None, resetStats = True):
        if resetStats:
            self.accuracy = 0.0
            self.recall = 0.0
            self.precision = 0.0
            self.f_measure = 0.0

            self._correctDeductionsCount = 0
            self._totalDeductionsCount = 0
            self._precision_per_class = np.zeros(len(self.LANGUAGES))
            self._recall_per_class = np.zeros(len(self.LANGUAGES))

            self.confusion_matrix = np.full((len(self.LANGUAGES), len(self.LANGUAGES)), 0.0)

            self._TP_per_class = np.zeros(len(self.LANGUAGES))
            self._FP_per_class = np.zeros(len(self.LANGUAGES))
            self._FN_per_class = np.zeros(len(self.LANGUAGES))
            self._TN_per_class = np.zeros(len(self.LANGUAGES))

            self._f_measure_per_class =  np.zeros(len(self.LANGUAGES))

        with open(fileName, encoding="utf8") as f:
            fil = f.read().splitlines()

            if not maxLine:
                for line in fil:
                    nb.guessTweetLanguage(line, newN=newN, newWeight=newWeight)
 
            else:
                for line in fil[:min(len(fil), maxLine)]:
                    nb.guessTweetLanguage(line,  newN=newN, newWeight=newWeight)


            self.accuracy = self._correctDeductionsCount/self._totalDeductionsCount

            # Calculate recall and precision per class

            for i in range(0, len(self.LanToNGrams)):
                self._recall_per_class[i] = self._TP_per_class[i]/(self._TP_per_class[i] + self._FP_per_class[i])
                self._precision_per_class[i] =  self._TP_per_class[i]/(self._TP_per_class[i] + self._FN_per_class[i])
            
            # Calculating F_measure per class

            for i in range(0, len(self.LanToNGrams)):
                self._f_measure_per_class[i] = (math.pow(self.beta,2) + 1) * self._recall_per_class[i] * self._precision_per_class[i]/(math.pow(self.beta,2)*self._precision_per_class[i] + self._recall_per_class[i])

            NanIndex = np.isnan(self._f_measure_per_class)
            self._f_measure_per_class[NanIndex] = 0.0
            self.precision = self._precision_per_class.mean()
            self.recall = self._recall_per_class.mean()
            self.f_measure = self._f_measure_per_class.mean()
            self.weighted_f_measure = np.average(self._f_measure_per_class, weights=self.probability_per_language)
            np.set_printoptions(precision=3, suppress=True)

            print("** Overall **")
            print("Accuracy: " + str(self.accuracy))
            print("Recall: " + str(self.recall))
            print("Precision: " + str(self.precision))
            print("F measure: " + str(self.f_measure))
            print("Weighted F measure: " + str(self.weighted_f_measure))
            print("** Confusion Matrix **")
            print("Legend: " + str(self.LANGUAGES))
            print(self.confusion_matrix)
            print("** Per class data **")
            print("Precision")
            print(self._precision_per_class)
            print("recall")
            print(self._recall_per_class)
            print("f-measure")
            print(self._f_measure_per_class)

if __name__ == "__main__":
    nb = NaiveBayerClassifier(N=3, V=4, weight=0.000000001, test=False)
    nb.trainFromTweets('training-tweets.txt')
    nb.runML('test-tweets-given.txt')