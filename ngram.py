
class HyperParameterOptimizer:
# Perform on discrete value sweep using different hyperparameter to optimize the ML model

    def __init__(self, *args, **kwargs):
        pass

class NGram:
    def __init__(self, *args, **kwargs):
        lan = kwargs.pop("lan")
        N = kwargs.pop("N")

class MetaStatistics:

    accuracy = 0
    recall = 0

class NaiveBayerClassifier(MetaStatistics):

    languages = ['eu', 'ca', 'gl', 'es', 'en', 't']


    def __init__(self, *args, **kwargs):
        N = kwargs.pop('N', 3)

        nGrams= {}

        for lan in self.languages:
            nGrams[lan] = NGram(lan = lan, N = self.N)



count = 0

with open('training-tweets.txt', encoding="utf8") as f:
    fil = f.read().splitlines()
    print(len(fil))
    print(fil[0].split("\t"))
