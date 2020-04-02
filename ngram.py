from collections import defaultdict
import math as math

from icu import LocalData


ISALPHA_VOCABULARY_SIZE = 116766 

def genAlphabetSet(V = 1):
    start = ord('a')
    end = ord('z')
    ans = set()
    for i in range(start, end+1):
        ans.add(chr(i))
    if V == 2:
        start = ord('A')
        end = ord('Z')
        for i in range(start, end+1):
            ans.add(chr(i))

    return ans

class NGramCell:
    dimension = 1
    parent = None

    def __init__(self, *args, **kwargs):
        self.dimension = kwargs.pop('dimension', 1)
        self.parent = kwargs.pop('parent', None)
        self.char = kwargs.pop('char', "Root")
        self.nextChar = {}
        self.count = 0

class NGram:
    def __init__(self, *args, **kwargs):
        self.lan = kwargs.pop("lan")
        self.N = kwargs.pop("N")
        self.V = kwargs.pop('V', 1)
        self.weight = kwargs.pop("weight", 0.001)
        self.n = self.N
        self.matrixRoot = NGramCell(dimension=0)
        self.denominator = None
        self.test = kwargs.pop("test", False)

        if self.V == 4:
            self.v = 3

        if self.V == 1 or self.V == 2:
            self.legalCharSet = genAlphabetSet(self.V)
            self.vocabulary = len(self.legalCharSet)

        if self.V == 3:
            self.vocabulary = ISALPHA_VOCABULARY_SIZE

        if self.V == 4:
            self.vocabulary = ISALPHA_VOCABULARY_SIZE + 1

    def addTrainingInput(self, txt):
        splicedInput = self.spliceAndCleanInput(txt)
        for element in splicedInput:
            for j in range(0, len(element)):
                if element[j] not in self.matrixRoot.nextChar:
                    self.matrixRoot.nextChar[element[j]] = NGramCell(char=element[j])
                currentMatrixCell = self.matrixRoot.nextChar[element[j]]
                for i in range(j, min(j+ self.n, len(element))):
                    currentMatrixCell.count += 1
                    if i + 1 < len(element):
                        if element[i+1] not in currentMatrixCell.nextChar:
                            currentMatrixCell.nextChar[element[i+1]] = NGramCell(char= element[i+1], dimension=currentMatrixCell.dimension+1, parent=currentMatrixCell)
                        currentMatrixCell = currentMatrixCell.nextChar[element[i+1]]
        if self.test:
            stack = []
            stack.append(self.matrixRoot)
            print("*** Traing Ngram for: ")
            print(splicedInput)
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


    
    def getProbability(self, txt, newN=None, newWeight=None):
        if newN != None:
            self.N = newN
        if newWeight != None:
            self.weight = newWeight
        splicedInput = self.spliceAndCleanInput(txt)
        totalProbability = 0.0
        self.denominator = None
        for element in splicedInput:
            for j in range(0, len(element)):
                if self.N == 1:
                    if element[j] not in self.matrixRoot.nextChar:
                        numerator = self.weight*1.0
                    else:
                        numerator = self.matrixRoot.nextChar[element[j]].count + self.weight*1.0
                    if self.denominator == None:
                        self.denominator = 0.0
                        for char in self.matrixRoot.nextChar:
                            self.denominator += self.firstChar.count
                    denominator = self.denominator + self.weight*math.pow(self.vocabulary, self.N)
                    probability = numerator/denominator
                    totalProbability += math.log10(probability)

                if self.N > 1:
                        currentMatrixCell = self.matrixRoot
                        for i in range(j, j+ self.n):
                            if i < len(element):
                                if  element[i] not in currentMatrixCell.nextChar:
                                    probability = self.weight*1.0/(self.weight*math.pow(self.vocabulary, self.N))
                                    totalProbability += math.log10(probability)
                                    currentMatrixCell = None
                                    break
                                else:
                                    currentMatrixCell = currentMatrixCell.nextChar[element[i]]
                            else:
                                currentMatrixCell = None
                                break
  
                        if currentMatrixCell != None:
                            #print(currentMatrixCell.dimension)
                            numerator = currentMatrixCell.count + self.weight*1.0
                            denominator = currentMatrixCell.parent.count + self.weight*math.pow(self.vocabulary, self.N)
                            probability = numerator/denominator

                            totalProbability += math.log10(probability)
                        
                        # numerator = 
        return totalProbability


    def spliceAndCleanInput(self, txt) -> [str]:
        ans = []
        if self.V == 1:
            txt = txt.lower()

        if self.V == 4:
            txt = txt.lower()
            txt = txt.split(" ")
            temp = []
            for w in txt:
                if w.startswith("@") or w.startswith("http"):
                    pass
                else:
                    temp.append(w)
            txt = "*".join(w for w in temp)


        start = 0

        for i in range (0, len(txt)):
            if (self.V == 1 or self.V == 2):
                if str(txt[i]) not in self.legalCharSet:
                    word = txt[start:i]

                    if len(word) != 0:
                        ans.append(txt[start:i])
                    
                    start = i+1

            if (self.V == 3):
                if not txt[i].isalpha():
                    word = txt[start:i]

                    if len(word) != 0:
                        ans.append(txt[start:i])
                    
                    start = i+1        

            if (self.V == 4):
                if not txt[i].isalpha() and txt[i] != "*":
                    word = txt[start:i]
                    if len(word) != 0:
                        ans.append(txt[start:i])
                    
                    start = i+1                           
        #ans.append(txt[start:len(txt)])
        
        #print(self.legalCharSet)
        #print(ans)
        return ans


class MetaStatistics:

    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    f_measure = 0.0
    beta = 1.0
    harmonic_mean = []

class NaiveBayerClassifier(MetaStatistics):

    LANGUAGES = ['eu', 'ca', 'gl', 'es', 'en', 't']


    def __init__(self, *args, **kwargs):
        self.N = kwargs.pop('N', 3)
        self.V = kwargs.pop('V', 1)
        self.weight = kwargs.pop("weight", 0.001)
        if "LANGUAGES" in kwargs:
            self.LANGUAGES = kwargs.pop('LANGUAGES')
    
        self.LanToNGrams= {}

        for lan in self.LANGUAGES:
            self.LanToNGrams[lan] = NGram(lan = lan, N = self.N, V =self.V, weight = self.weight)

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

    def trainFromTweets(self, fileName, maxLine = None):
        with open(fileName, encoding="utf8") as f:
            fil = f.read().splitlines()

            if not maxLine:
                for line in fil:
                    self.addTrainingInputLine(line)
            else:
                for line in fil[:min(len(fil), maxLine)]:
                    self.addTrainingInputLine(line)


    def guessTweetLanguage(self, stringLine, logFile= None, newN= None, newWeight = None):
        lineList = stringLine.split("\t")
        id = lineList[0]
        correctLan = lineList[2]
        txt = lineList[3]
        #print(id)
        #print(correctLan)
        #print(txt)

        guesses = []

        for lan in self.LanToNGrams:
            guesses.append([lan, self.LanToNGrams[lan].getProbability(txt, newN, newWeight)])

        
        guesses.sort(key= lambda x: x[1], reverse = True)


        printStr = id + "  " + correctLan + "  " + str(guesses[0][1]) + "  " + guesses[0][0] + "  "

        if guesses[0][0] == correctLan:
            printStr += "correct"
        else:
            printStr += "wrong"
        self.precision += 1
        if correctLan != guesses[0][0]:
            #print(guesses)
            self.recall += 1
            #print(printStr)


    def runML(self, fileName, newN= None, newWeight = None, maxLine = None, resetStats = True):
        if resetStats:
            self.accuracy = 0.0
            self.recall = 0.0
            self.precision = 0.0
            self.f_measure = 0.0
            self.beta = 1.0
            self.harmonic_mean = []

        with open(fileName, encoding="utf8") as f:
            fil = f.read().splitlines()

            if not maxLine:
                for line in fil:
                    try:
                        nb.guessTweetLanguage(line, newN=newN, newWeight=newWeight)
                    except Exception as e:
                        print(e)
            else:
                for line in fil[:min(len(fil), maxLine)]:
                    try:
                        nb.guessTweetLanguage(line,  newN=newN, newWeight=newWeight)
                    except Exception as e:
                        print(e)
            print(self.recall)
            print(self.precision)

class HyperParameterOptimizer:
# Perform on discrete value sweep using different hyperparameter to optimize the ML model

    def __init__(self, *args, **kwargs):
        pass

def processLine(stringLine):
    lineList = stringLine.split("\t")
    id = lineList[0]
    correctLan = lineList[2]
    txt = lineList[3]

    #print(id)
    #print(correctLan)
    #print(txt)

# print(len(genAlphabetSet()))
if __name__ == "__main__":
    nb = NaiveBayerClassifier(N=3, V=4, weight=0.000005, test=False)
    nb.trainFromTweets('training-tweets.txt')
    nb.runML('test-tweets-given.txt')