from collections import defaultdict
import numpy as np
import math as math
from util import genAlphabetSet, ISALPHA_VOCABULARY_SIZE, V4_OPTIMAL_WEIGHT

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
        self.V = kwargs.pop('V', 1)
        self.weight = kwargs.pop("weight", 0.001)
        self.matrixRoot = NGramCell(dimension=0)
        self.denominator = None
        self.test = kwargs.pop("test", False)

        if self.V == 4:
            self.weight = V4_OPTIMAL_WEIGHT

        if self.V == 1 or self.V == 2 or self.V == 4:
            self.legalCharSet = genAlphabetSet(self.V)
            self.vocabulary = len(self.legalCharSet)

        if self.V == 3:
            self.vocabulary = ISALPHA_VOCABULARY_SIZE

    def addTrainingInput(self, txt):
        splicedInput = self.spliceAndCleanInput(txt)
        for element in splicedInput:
            for j in range(0, len(element)):
                if element[j] not in self.matrixRoot.nextChar:
                    self.matrixRoot.nextChar[element[j]] = NGramCell(char=element[j])
                currentMatrixCell = self.matrixRoot.nextChar[element[j]]
                for i in range(j, min(j+ self.N, len(element))):
                    currentMatrixCell.count += 1
                    if i + 1 < len(element):
                        if element[i+1] not in currentMatrixCell.nextChar:
                            currentMatrixCell.nextChar[element[i+1]] = NGramCell(char= element[i+1], dimension=currentMatrixCell.dimension+1, parent=currentMatrixCell)
                        currentMatrixCell = currentMatrixCell.nextChar[element[i+1]]
        '''
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
        '''


    
    def getProbability(self, txt, newN=None, newWeight=None):
        if newN != None:
            self.N = newN
        if newWeight != None:
            self.weight = newWeight
        splicedInput = self.spliceAndCleanInput(txt)
        totalProbability = 0.0
        #print(splicedInput)
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
                            self.denominator += self.matrixRoot.count
                    denominator = self.denominator + self.weight*math.pow(self.vocabulary, self.N)
                    probability = numerator/denominator
                    totalProbability += math.log10(probability)

                if self.N > 1:
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
                            #print(currentMatrixCell.dimension)
                            #print("********")
                            
                            #print(currentMatrixCell.char + "  " + str(currentMatrixCell.count))
                            #print(currentMatrixCell.parent.char + "  " + str(currentMatrixCell.parent.count))
                            #print(currentMatrixCell.parent.parent.char + "  " + str(currentMatrixCell.parent.parent.count))

                            numerator = currentMatrixCell.count + self.weight*1.0
                            denominator = currentMatrixCell.parent.count + self.weight*math.pow(self.vocabulary, self.N)
                            probability = numerator/denominator

                            totalProbability += math.log10(probability)
                        
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
                if w.startswith("@") or w.startswith("http") or w.startswith("#"):
                    pass
                else:
                    temp.append(w)
            txt = " ".join(w for w in temp)


        start = 0

        for i in range (0, len(txt)):
            if (self.V == 1 or self.V == 2 or self.V == 4):
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
        #ans.append(txt[start:len(txt)])
        
        #print(self.legalCharSet)
        #print(ans)
        return ans