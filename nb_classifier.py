from collections import defaultdict
import numpy as np
import math as math
import sys
from util import V3_OPTIMAL_WEIGHT
from ngram import NGram

class MetaStatistics:

    LANGUAGES = ['eu', 'ca', 'gl', 'es', 'en', 'pt']

    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    f_measure = 0.0
    weighted_f_measure = 0.0
    beta = 1.0

    _correctDeductionsCount = 0
    _totalDeductionsCount = 0

    def __init__(self, *args, **kwargs):
        if "LANGUAGES" in kwargs:
            self.LANGUAGES = kwargs.pop('LANGUAGES')
        self.beta = kwargs.pop("beta", 1)
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

class NaiveBayerClassifier(MetaStatistics):

    def __init__(self, *args, **kwargs):
        super(NaiveBayerClassifier, self).__init__(*args, **kwargs)
        self.N = kwargs.pop('N', 3)
        self.V = kwargs.pop('V', 0)
        self.weight = kwargs.pop("weight", 0.001)
    
        self.LanToNGrams= {}
        self.LanToIndex = {}

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

        '''
        if self.test:
            for lan in self.LanToNGrams:
                stack = []
                stack.append(self.LanToNGrams[lan].matrixRoot)
                print("*** Trainingg Ngram for: " + lan)
                while len(stack) !=0:
                    current = stack.pop()
                    printStr = ""
                    for i in range (0, current.dimension):
                        printStr += "\t"
                    printStr += current.char
                    printStr += " count: " + str(current.count)
                    print(printStr)
                    for element in current.nextChar:
                        if current.nextChar[element].dimension <= self.N:
                            stack.append(current.nextChar[element])
        '''


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
        score = "{:.2e}".format(guesses[0][1])
        printStr = id + "  " + correctLan + "  " + score + "  " + guessLan + "  "

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
        
        if self.test:
            logFile.write(printStr + '\n')

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

        if self.test:
            if self.V == 3:
                traceFile = open("trace_myModel.txt", "w", encoding="utf-8")
                evalFile = open("eval_myModel.txt", "w", encoding="utf-8")
            else : 
                traceFile = open("trace_"+ str(self.V) + "_" + str(self.N) + "_" + str(self.weight) + ".txt", "w", encoding="utf-8")
                evalFile = open("eval_"+ str(self.V) + "_" + str(self.N) + "_" + str(self.weight) + ".txt", "w", encoding="utf-8")
        else:
            traceFile = None
            evalFile = None
            
        with open(fileName, encoding="utf8") as f:
            fil = f.read().splitlines()

            if not maxLine:
                for line in fil:
                    self.guessTweetLanguage(line, traceFile, newN=newN, newWeight=newWeight)
 
            else:
                for line in fil[:min(len(fil), maxLine)]:
                    self.guessTweetLanguage(line, traceFile, newN=newN, newWeight=newWeight)


            self.accuracy = self._correctDeductionsCount/self._totalDeductionsCount

            # Calculate recall and precision per class

            for i in range(0, len(self.LanToNGrams)):
                denominator = (self._TP_per_class[i] + self._FP_per_class[i])
                if denominator != 0.0:
                    self._recall_per_class[i] = self._TP_per_class[i]/(self._TP_per_class[i] + self._FP_per_class[i])
                self._precision_per_class[i] =  self._TP_per_class[i]/(self._TP_per_class[i] + self._FN_per_class[i])
            
            # Calculating F_measure per class

            for i in range(0, len(self.LanToNGrams)):
                denominator = (math.pow(self.beta,2)*self._precision_per_class[i] + self._recall_per_class[i])
                if denominator != 0.0:
                    self._f_measure_per_class[i] = (math.pow(self.beta,2) + 1) * self._recall_per_class[i] * self._precision_per_class[i]/(math.pow(self.beta,2)*self._precision_per_class[i] + self._recall_per_class[i])

            NanIndex = np.isnan(self._f_measure_per_class)
            self._f_measure_per_class[NanIndex] = 0.0
            self.precision = self._precision_per_class.mean()
            self.recall = self._recall_per_class.mean()
            self.f_measure = self._f_measure_per_class.mean()
            self.weighted_f_measure = np.average(self._f_measure_per_class, weights=self.probability_per_language)
            np.set_printoptions(precision=4, suppress=True, legacy='1.13')
            

            if self.test:
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
                
                evalFile.write(str(round(self.accuracy, 4)) + '\n')
                str_precision_per_class = np.array2string(self._precision_per_class, precision=4, separator='  ', formatter={'float_kind':lambda x: "%.4f" % x})
                evalFile.write(str_precision_per_class[1:-1]+ '\n')
                str_recall_per_class = np.array2string(self._recall_per_class, precision=4, separator='  ', formatter={'float_kind':lambda x: "%.4f" % x})
                evalFile.write(str_recall_per_class[1:-1]+ '\n')
                str_f_measure_per_class = np.array2string(self._f_measure_per_class, precision=4, separator='  ', formatter={'float_kind':lambda x: "%.4f" % x})
                evalFile.write(str_f_measure_per_class[1:-1]+ '\n')
                evalFile.write(str(round(self.f_measure, 4)) + '  ')
                evalFile.write(str(round(self.weighted_f_measure, 4)))   
                evalFile.close()

            return {"f_measure": self.f_measure, 
                    "weighted_f_measure": self.weighted_f_measure, 
                    "accuracy": self.accuracy, 
                    "recall": self.recall, 
                    "precision": self.precision, 
                    "confusion_matrix": self.confusion_matrix,
                    "precision_per_class": self._precision_per_class,
                    "recall_per_class": self._recall_per_class,
                    "f_measure_per_class": self._f_measure_per_class}


def main(*args):
    key = {}
    for arg in args:
        k = arg.split("=")[0]
        if k == "D":
            key[k] = float(arg.split("=")[1])
        elif k == "TRAIN" or k == "TEST" :
            key[k] = arg.split("=")[1]
        else:
            key[k] = int(arg.split("=")[1])

    if key["V"] == 3:
        nb = NaiveBayerClassifier(N=3, V=3, weight= V3_OPTIMAL_WEIGHT,test=True)
    else:
        nb = NaiveBayerClassifier(N=key["N"], V=key["V"], weight=key["D"], test=True)
        
    if "TRAIN" in key: 
        nb.trainFromTweets(key["TRAIN"])
    else:
        nb.trainFromTweets('training-tweets.txt')
    if "TEST" in key: 
        nb.runML(key["TEST"])
    else:
        nb.runML('test-tweets-given.txt')

if __name__ == "__main__":
    if len(sys.argv) ==2:
        main(sys.argv[1])
    elif len(sys.argv) ==3:
        main(sys.argv[1], sys.argv[2])
    elif  len(sys.argv) ==4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif  len(sys.argv) ==5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif  len(sys.argv) ==6:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        raise SyntaxError("Exceed maximum arguments")