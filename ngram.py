from collections import defaultdict


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
        self.nextChar = {}
        self.count = 0

class NGram:
    def __init__(self, *args, **kwargs):
        self.lan = kwargs.pop("lan")
        self.N = kwargs.pop("N")
        self.V = kwargs.pop('V', 1)
        self.weight = kwargs.pop("weight", 0)
        self.v = self.V
        self.firstChar = {}
        if self.V == 4:
            self.v = 3

        if self.V == 1 or self.V == 2:
            self.legalCharSet = genAlphabetSet(self.V)

    def addTrainingInput(self, txt):
        splicedInput = self.spliceAndCleanInput(txt)
        input = []
        for element in splicedInput:
            for j in range(0, len(element)):
                stack = []
                if element[j] not in self.firstChar:
                    self.firstChar[element[j]] = NGramCell()
                stack.append(self.firstChar[element[j]])
                for i in range(j, min(j+ self.v, len(element))):
                    stack[-1].count += 1

                    if i + 1 < len(element):
                        if element[i+1] not in stack[-1].nextChar:
                            stack[-1].nextChar[element[i+1]] = NGramCell(dimension=stack[-1].dimension+1, parent=stack[-1])
                        stack.append(stack[-1].nextChar[element[i+1]])

        '''
        for elem in self.firstChar:
            print("************")
            print(elem)
            print(self.firstChar[elem].count)

            for item in self.firstChar[elem].nextChar:
                print("!!!!!")
                print(item)
                print(self.firstChar[elem].nextChar[item].count)
                print("!!")
                for i in self.firstChar[elem].nextChar[item].nextChar:
                    print(i)
                    print(self.firstChar[elem].nextChar[item].nextChar[i].count)
                print("!!!!")
            print("************")
        
        return
        '''

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
        print(ans)
        return ans


class MetaStatistics:

    accuracy = 0
    recall = 0

class NaiveBayerClassifier(MetaStatistics):

    LANGUAGES = ['eu', 'ca', 'gl', 'es', 'en', 't']


    def __init__(self, *args, **kwargs):
        self.N = kwargs.pop('N', 3)
        self.V = kwargs.pop('V', 1)
        if "LANGUAGES" in kwargs:
            self.LANGUAGES = kwargs.pop('LANGUAGES')
    
        self.LanToNGrams= {}

        for lan in self.LANGUAGES:
            self.LanToNGrams[lan] = NGram(lan = lan, N = self.N, V =self.V)

    def processInputLine(self, stringLine, training=True):
        lineList = stringLine.split("\t")
        id = lineList[0]
        correctLan = lineList[2]
        txt = lineList[3]

        print(id)
        print(correctLan)
        print(txt)

        if training:
            if correctLan in self.LanToNGrams:
                self.LanToNGrams[correctLan].addTrainingInput(txt)


class HyperParameterOptimizer:
# Perform on discrete value sweep using different hyperparameter to optimize the ML model

    def __init__(self, *args, **kwargs):
        pass

def processLine(stringLine):
    lineList = stringLine.split("\t")
    id = lineList[0]
    correctLan = lineList[2]
    txt = lineList[3]

    print(id)
    print(correctLan)
    print(txt)

# print(len(genAlphabetSet()))
nb = NaiveBayerClassifier(V=3)
with open('training-tweets.txt', encoding="utf8") as f:
    fil = f.read().splitlines()

    for line in fil[:1]:
        nb.processInputLine(line)
string = 'E'
#print(string.isalpha())