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


class NGram:
    def __init__(self, *args, **kwargs):
        self.lan = kwargs.pop("lan")
        self.N = kwargs.pop("N")
        self.V = kwargs.pop('V', 1)
        self.v = self.V
        if self.V == 4:
            self.v = 3

        if self.V == 1 or self.V == 2:
            self.legalCharSet = genAlphabetSet(self.V)

    def addTrainingInput(self, txt):
        splicedInput = self.spliceAndCleanInput(txt)
        input = []
        for element in splicedInput:
            for i in range(0, len(element)):
                if i+ self.v <= len(element):
                    print(element[i:i+self.v])
        
        return

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
nb = NaiveBayerClassifier(V=2)
with open('training-tweets.txt', encoding="utf8") as f:
    fil = f.read().splitlines()

    for line in fil[:20]:
        nb.processInputLine(line)
string = 'E'
#print(string.isalpha())