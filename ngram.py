from collections import defaultdict
import numpy as np
import math as math
from util import genAlphabetSet, ISALPHA_VOCABULARY_SIZE, V3_OPTIMAL_WEIGHT

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
        self.N = kwargs.pop("N", 3)
        self.V = kwargs.pop('V', 0)
        self.weight = kwargs.pop("weight", 0.001)
        self.matrixRoot = NGramCell(dimension=0)
        self.denominator = None
        self.test = kwargs.pop("test", False)

        if self.V == 3:
            self.weight = V3_OPTIMAL_WEIGHT

        if self.V == 0 or self.V == 1 or self.V == 3:
            self.legalCharSet = genAlphabetSet(self.V)
            self.vocabulary = len(self.legalCharSet)

        if self.V == 2:
            self.vocabulary = ISALPHA_VOCABULARY_SIZE

    def addTrainingInput(self, txt):
        splicedInput = self.spliceAndCleanInput(txt)
        for element in splicedInput:
            for j in range(0, len(element)):
                if element[j] not in self.matrixRoot.nextChar:
                    self.matrixRoot.nextChar[element[j]] = NGramCell(char=element[j], parent=self.matrixRoot)
                currentMatrixCell = self.matrixRoot.nextChar[element[j]]
                self.matrixRoot.count +=1
                for i in range(j, min(j+ self.N, len(element))):
                    currentMatrixCell.count += 1
                    if i + 1 < len(element):
                        if element[i+1] not in currentMatrixCell.nextChar:
                            currentMatrixCell.nextChar[element[i+1]] = NGramCell(char= element[i+1], dimension=currentMatrixCell.dimension+1, parent=currentMatrixCell)
                        currentMatrixCell = currentMatrixCell.nextChar[element[i+1]]
    
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
                if self.N >= 1:
                    currentMatrixCell = self.matrixRoot
                    for i in range(j, j+ self.N):
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

                        numerator = currentMatrixCell.count + self.weight*1.0
                        denominator = currentMatrixCell.parent.count + self.weight*math.pow(self.vocabulary, self.N)
                        probability = numerator/denominator

                        totalProbability += math.log10(probability)
                        
        return totalProbability


    def spliceAndCleanInput(self, txt) -> [str]:
        ans = []
        if self.V == 0:
            txt = txt.lower()

        if self.V == 3:
            txt = txt.lower()
            txt = txt.split(" ")
            temp = []
            for w in txt:
                if w.startswith("@") or w.startswith("http") or w.startswith("#"):
                    pass
                else:
                    temp.append(w)
            txt = " ".join(w for w in temp)


        start = 0

        for i in range (0, len(txt)):
            if (self.V == 0 or self.V == 1 or self.V == 3):
                if str(txt[i]) not in self.legalCharSet:
                    word = txt[start:i]

                    if len(word) != 0:
                        ans.append(txt[start:i])
                    
                    start = i+1

            if (self.V == 2):
                if not txt[i].isalpha():
                    word = txt[start:i]

                    if len(word) != 0:
                        ans.append(txt[start:i])
                    
                    start = i+1                               

        return ans